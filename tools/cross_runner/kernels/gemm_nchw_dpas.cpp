#include <cm/cm.h>
#include <cm/cmtl.h>

#define HALF half
#define FLOAT float
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2

#define SIZE_OF_HF16_BYTE 2
#define SIZE_PER_DPAS_HF16 128  // DPAS works for half float matrix [8x16] [16x8]

#define ODD_M 0

_GENX_ inline void myDPAS8(matrix_ref<HALF, 8, 16> matA,
                            matrix_ref<HALF, 8, 16> matB,
                            matrix_ref<FLOAT, 8, 8> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

extern "C" _GENX_MAIN_ void gemm_nchw_dpas(
	SurfaceIndex surface_input_a [[type("buffer_t")]],
	SurfaceIndex surface_input_b [[type("buffer_t")]],
#if USE_INPUTC
	SurfaceIndex surface_input_c [[type("buffer_t")]],
#endif
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
	// use byte to calcute offset
	uint gidX = cm_group_id(DIM_X);
	uint gidY = cm_group_id(DIM_Y);
	uint gidZ = cm_group_id(DIM_Z);
    uint tidX = cm_local_id(DIM_X);
    uint tidY = cm_local_id(DIM_Y);
	uint tidZ = cm_local_id(DIM_Z);

	const uint32_t thread_id_0 = gidX * cm_local_size(DIM_X) + tidX;
	const uint32_t thread_id_1 = gidY * cm_local_size(DIM_Y) + tidY;
	const uint32_t thread_id_2 = gidZ * cm_local_size(DIM_Z) + tidZ;
	const uint32_t base_offset_a =  thread_id_0 * TILE_M * SIZE_K * SIZE_OF_HF16_BYTE;
	const uint32_t base_offset_b =  thread_id_1 * TILE_N * SIZE_OF_HF16_BYTE;
	const uint32_t base_offset_output =  (thread_id_0 * TILE_M * SIZE_N + thread_id_1 * TILE_N) * SIZE_OF_HF16_BYTE;

#if ODD_M
	bool bLAST_M_STEP = (gidX == (SIZE_M/TILE_M-1) ) ? true : false;
	uint32_t bLAST_M_LEFTOVER = SIZE_M % TILE_M;
	bool bLOAD_LEFTOVER = (bLAST_M_STEP && bLAST_M_LEFTOVER) ? true : false;
#endif
	// init TILE_A
	vector<HALF, TILE_M * TILE_K > readA = 0.0; 	// A tile: (8*TILE_M/8)M x 16K
	matrix_ref<HALF, TILE_M, TILE_K> readA_m = readA.format<HALF, TILE_M, TILE_K>();

	// init TILE_B
	vector<HALF, SIZE_PER_DPAS_HF16> readB = 0.0;  //B tile: 16Kx8N
	matrix_ref<HALF, 8, 16> readB_m = readB.format<HALF, 8, 16>();
	// TILE_B: read two lines, and ordered into DPAS required format
	matrix<HALF, TILE_K/2, TILE_N> rowX2_0 = 0.0;  
	matrix<HALF, TILE_K/2, TILE_N> rowX2_1 = 0.0;
	
	//init the accumulators
	matrix<FLOAT, TILE_M, TILE_N> result1 = 0.0; 
	matrix<FLOAT, TILE_M, TILE_N> result1_last = 0.0;  
	
	for( int step = 0; step < SIZE_K; step += TILE_K)
	{
		const uint32_t step_base_offset_a = base_offset_a + step * SIZE_OF_HF16_BYTE;
		const uint32_t step_base_offset_b = base_offset_b + (step / TILE_K) * SIZE_N * TILE_K * SIZE_OF_HF16_BYTE;
		
		//cache elements in matrix B 
		#pragma unroll
		for(int row = 0; row < TILE_K/2; row++)
		{
			const uint32_t rowX2 = row * 2;
			const uint32_t read_offset_b = step_base_offset_b + (rowX2 * SIZE_N)* SIZE_OF_HF16_BYTE;
			rowX2_0.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);  
			rowX2_1.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + SIZE_N* SIZE_OF_HF16_BYTE);  
		}		

		#pragma unroll
		for(int m=0; m < TILE_M/8; m++)
		{
			//cache elements in matrix A
			#pragma unroll
			for(int row = 0; row < 8; row++)
			{
				const unsigned read_offset_a = step_base_offset_a + (row * SIZE_K) * SIZE_OF_HF16_BYTE;
				// Read from inputs surfaces row M x 16K
				readA_m.select<1,1,TILE_K,1>(row + m * 8, 0).format<U32>() = cm_load<U32, TILE_K/2 , DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, read_offset_a + SIZE_K * (8 * m) * SIZE_OF_HF16_BYTE);
			}	
			
			//calcute DPAS
			#pragma unroll	
			for(int n = 0; n < TILE_N/8; n++)  
			{
				#pragma unroll	
				for(int k = 0; k < TILE_K/16; k++)
				{
					#pragma unroll
					for (int row = 0; row < 8; row++)
					{		
						readB_m.select<1,1,8,2>(row, 0)= rowX2_0.select<1,1,8,1>(row + k*8, 8*n);
						readB_m.select<1,1,8,2>(row, 1)= rowX2_1.select<1,1,8,1>(row + k*8, 8*n);	
					}	
					myDPAS8(readA_m.select<8,1,16,1>(m * 8, k*16),  readB_m, result1.select<8,1,8,1>(m * 8,n * 8));  
				}
			}
		}
#if ODD_M
		if(bLOAD_LEFTOVER)
		{
			
			readA_m = 0.0; 
			//cache the last elements in matrix A
			#pragma unroll
			for(int row = 0; row < bLAST_M_LEFTOVER; row++)
			{
				const unsigned read_offset_a = step_base_offset_a + (TILE_M * SIZE_K) * SIZE_OF_HF16_BYTE + (row * SIZE_K) * SIZE_OF_HF16_BYTE;
				// Read from inputs surfaces row M x 16K
				readA_m.select<1,1,TILE_K,1>(row , 0).format<U32>() = cm_load<U32, TILE_K/2 , DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, read_offset_a);
			}

			for(int m= 0; m < TILE_M/8; m++)
			{		
				//calcute DPAS
				#pragma unroll	
				for(int n = 0; n < TILE_N/8; n++)  
				{
					#pragma unroll	
					for(int k = 0; k < TILE_K/16; k++)
					{
						#pragma unroll
						for (int row = 0; row < 8; row++)
						{		
							readB_m.select<1,1,8,2>(row, 0)= rowX2_0.select<1,1,8,1>(row + k*8, 8*n);
							readB_m.select<1,1,8,2>(row, 1)= rowX2_1.select<1,1,8,1>(row + k*8, 8*n);	
						}	
						myDPAS8(readA_m.select<8,1,16,1>(m * 8, k*16),  readB_m, result1_last.select<8,1,8,1>(m * 8,n * 8));  
					}
				}
			}

			if(step ==0 && thread_id_1 == 1)
			{
				printf(" readA: SIZE_M:%d SIZE_N:%d\n",SIZE_M,SIZE_N);
				for (int i = 0; i < TILE_M; i++)
				{
					printf(" row%d", i);
					for (int j = 0; j < TILE_K;j++)
					{ 
						printf(" %f",readA(i *TILE_K+j));
					}
					printf("\n");\
				}	

				printf(" readB: SIZE_M:%d SIZE_N:%d\n",SIZE_M,SIZE_N);
				for (int i = 0; i < 8; i++)
				{
					printf(" row%d", i);
					for (int j = 0; j < 16;j++)
					{ 
						printf(" %f",readB(i *16+j));
					}
					printf("\n");\
				}

				printf(" result: SIZE_M:%d SIZE_N:%d\n",SIZE_M,SIZE_N);
				for (int i = 0; i < TILE_M; i++)
				{
					printf(" row%d", i);
					for (int j = 0; j < TILE_N;j++)
					{ 
						printf(" %f",result1_last(i ,j));
					}
					printf("\n");\
				}	
			}
			
		}
#endif
	}

	// if (thread_id_0 == 0 && thread_id_1 == 0 )
	// {
	// 	printf(" normal_readA: SIZE_M:%d SIZE_N:%d\n",SIZE_M,SIZE_N);
	// 	for (int i = 0; i < TILE_M; i++)
	// 	{
	// 		printf(" row%d", i);
	// 		for (int j = 0; j < TILE_K;j++)
	// 		{ 
	// 			printf(" %f",readA(i *TILE_K+j));
	// 		}
	// 		printf("\n");\
	// 	}

	// 	printf(" normal_readB: SIZE_M:%d SIZE_N:%d\n",SIZE_M,SIZE_N);
	// 	for (int i = 0; i < 8; i++)
	// 	{
	// 		printf(" row%d", i);
	// 		for (int j = 0; j < 16;j++)
	// 		{ 
	// 			printf(" %f",readB(i *16+j));
	// 		}
	// 		printf("\n");\
	// 	}
	// }

	vector<HALF, TILE_N> result_hf16_CL1 = 0.0;
	result1 *= HALF(ALPHA);

#if USE_INPUTC
	matrix<HALF, TILE_M, TILE_N> bias = 0.0; 
	#pragma unroll
	for(int i = 0; i < TILE_M; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_HF16_BYTE;
		bias.select<1,1, TILE_N, 1>(i,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_c, write_index);
	}
	result1 += HALF(BETA)*bias;
#endif

	#pragma unroll
	for(int i = 0; i < TILE_M; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_HF16_BYTE;
		result_hf16_CL1.select<TILE_N, 1>(0)  = result1.select<1, 1, TILE_N, 1>(i, 0);
		cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
	}

#if ODD_M
	if(bLOAD_LEFTOVER)
	{
		#pragma unroll
		for(int i = 0; i < bLAST_M_LEFTOVER; i++)
		{
			const unsigned write_index = base_offset_output + TILE_M * SIZE_N * SIZE_OF_HF16_BYTE + i * SIZE_N * SIZE_OF_HF16_BYTE;	
			result_hf16_CL1.select<TILE_N, 1>(0)  = result1_last.select<1, 1, TILE_N, 1>(i, 0);
			cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
		}
	}
#endif

}
