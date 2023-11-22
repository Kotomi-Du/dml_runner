#include <cm/cm.h>
#include <cm/cmtl.h>

#define HALF half
#define FLOAT float
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2

#define SIZE_OF_HF16_BYTE 2
#define SIZE_PER_DPAS_HF16 128  // DPAS works for half float matrix [8x16] [16x8]

// SLM optimization
#define SLM_KN_SHARING 0

_GENX_ inline void myDPAS8(matrix_ref<HALF, 8, 16> matA,
                            matrix_ref<HALF, 8, 16> matB,
                            matrix_ref<FLOAT, 8, 8> result)
{
	result = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, 8>(result.format<FLOAT>(), matB.format<U32>(), matA.format<U32>());
}

extern "C" _GENX_MAIN_ void gemm_nchw_dpas(
	SurfaceIndex surface_input_a [[type("buffer_t")]],
	SurfaceIndex surface_input_b [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
#if !defined(EMPTY)

	// use byte to calcute offset
	uint gidX = cm_group_id(DIM_X);
	uint gidY = cm_group_id(DIM_Y);
	uint gidZ = cm_group_id(DIM_Z);
    uint tidX = cm_local_id(DIM_X);
    uint tidY = cm_local_id(DIM_Y);
	uint tidZ = cm_local_id(DIM_Z);

	const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	const uint32_t base_offset_a =  thread_id_0 * TILE_M * SIZE_K * SIZE_OF_HF16_BYTE;
	const uint32_t base_offset_b =  thread_id_1 * TILE_N * SIZE_OF_HF16_BYTE;
	const uint32_t base_offset_output =  (thread_id_0 * TILE_M * SIZE_N + thread_id_1 * TILE_N) * SIZE_OF_HF16_BYTE;
    
	//printf("%d,%d,%d\n", thread_id_0, thread_id_1, thread_id_2);
	// init TILE_A
	vector<HALF, SIZE_PER_DPAS_HF16 * (TILE_M / 8) * (TILE_K / 16) > readA = 0.0; 	// M=0..7,  K=0..15		// A tile: (8*TILE_M/8)M x 16K
	matrix_ref<HALF, TILE_M, TILE_K> readA_m = readA.format<HALF, TILE_M, TILE_K>();

	// init TILE_B
#if SLM_KN_SHARING
	cm_slm_init(TILE_K * TILE_N * SIZE_OF_HF16_BYTE); 
    uint slm_buffer = cm_slm_alloc(TILE_K * TILE_N * SIZE_OF_HF16_BYTE);
	vector<HALF, SIZE_PER_DPAS_HF16 * (TILE_N / 8) *(TILE_K/16)> readB = 0.0; 	// N=0..7,  K=0..15		//B tile: 16Kx8N
	matrix_ref<HALF, TILE_K/2, TILE_N*2> readB_m = readB.format<HALF, TILE_K/2, TILE_N*2>();	
#else
	vector<HALF, SIZE_PER_DPAS_HF16> readB = 0.0; 	// N=0..7,  K=0..15		//B tile: 16Kx8N
	matrix_ref<HALF, 8, 16> readB_m = readB.format<HALF, 8, 16>();
#endif

	
	//init the accumulators
	matrix<FLOAT, TILE_M, TILE_N> result1 = 0.0;  
	matrix_ref<FLOAT, TILE_M, TILE_N> result1ref = result1;
	
	for( int step = 0; step < SIZE_K; step += TILE_K)
	{
		const uint32_t step_base_offset_a = base_offset_a + step * SIZE_OF_HF16_BYTE;
		const uint32_t step_base_offset_b = base_offset_b + (step / TILE_K) * SIZE_N * TILE_K * SIZE_OF_HF16_BYTE;
		// TILE_B: read two lines, and ordered into DPAS required format
#if SLM_KN_SHARING	
		const uint32_t read_offset_b = step_base_offset_b + cm_local_id(0) * SIZE_N * 2 * SIZE_OF_HF16_BYTE;
		vector<uint32_t, TILE_N/2> packed_rowX2_0;
		vector<uint32_t, TILE_N/2> packed_rowX2_1;
		
		packed_rowX2_0.select<TILE_N/2,1>(0) = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b );  
		packed_rowX2_1.select<TILE_N/2,1>(0) = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + SIZE_N * SIZE_OF_HF16_BYTE);  

        // cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
		// cm_barrier();
		const uint32_t slm_write_base_offset = cm_local_id(0) * TILE_N * 2 * SIZE_OF_HF16_BYTE;
		cm_store_slm<uint32_t, TILE_N/2>(slm_write_base_offset, packed_rowX2_0.select<TILE_N/2, 1>(0));
		cm_store_slm<uint32_t, TILE_N/2>(slm_write_base_offset + TILE_N * SIZE_OF_HF16_BYTE, packed_rowX2_1.select<TILE_N/2, 1>(0));
		cm_slm_fence(CM_GLOBAL_COHERENT_FENCE);
		cm_barrier();

		#pragma unroll
		for (int row = 0; row < TILE_K; row++)
		{
			const uint32_t slm_read_base_offset = row * TILE_N * 2 * SIZE_OF_HF16_BYTE;
			readB_m.select<1,1,TILE_N,2>(row, 0).format<U32>()= cm_load_slm<uint32_t, TILE_N/2>(slm_read_base_offset);
			readB_m.select<1,1,TILE_N,2>(row, 1).format<U32>()= cm_load_slm<uint32_t, TILE_N/2>(slm_read_base_offset + TILE_N * SIZE_OF_HF16_BYTE);
		}
#else

		matrix<HALF, TILE_K/2, TILE_N> rowX2_0 = 0.0;  
		matrix<HALF, TILE_K/2, TILE_N> rowX2_1 = 0.0;

		//cache elements in matrix B 
		#pragma unroll
		for(int row = 0; row < TILE_K/2; row++)
		{
			const uint32_t rowX2 = row * 2;
			const uint32_t read_offset_b = step_base_offset_b + (rowX2 * SIZE_N)* SIZE_OF_HF16_BYTE;
			rowX2_0.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);  
			rowX2_1.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + SIZE_N* SIZE_OF_HF16_BYTE);  
		}		
#endif


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
					#if SLM_KN_SHARING
						myDPAS8(readA_m.select<8,1,16,1>(m * 8, k*16), readB_m.select<8,1,16,1>(k*8, n*16),result1ref.select<8,1,8,1>(m * 8,n * 8));
					#else
						#pragma unroll
						for (int row = 0; row < 8; row++)
						{		
							readB_m.select<1,1,8,2>(row, 0)= rowX2_0.select<1,1,8,1>(row + k*8, 8*n);
							readB_m.select<1,1,8,2>(row, 1)= rowX2_1.select<1,1,8,1>(row + k*8, 8*n);	
						}	
						myDPAS8(readA_m.select<8,1,16,1>(m * 8, k*16),  readB_m, result1ref.select<8,1,8,1>(m * 8,n * 8));  
					#endif
				}
			}
		}

		// if (thread_id_0 == 11 && thread_id_1 == 0)
		// {
		// 	for (int i = 0; i < TILE_M; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < TILE_N;j++)
        // 		{ 
        // 		    printf(" %f", result1ref(i , j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }
		
		// if (thread_id_0 == 11 && thread_id_1 == 0)
		// {
		// 	for (int i = 0; i < 16; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < TILE_K;j++)
        // 		{ 
        // 		    printf(" %f",readA(i *TILE_K+j));
		// 			// printf(" %f",readB(i *16+j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }

	}

	vector<HALF, TILE_N> result_hf16_CL1 = 0.0;
	result1 *= HALF(ALPHA);
	

	#pragma unroll
	for(int i = 0; i < TILE_M; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_HF16_BYTE;	
		result_hf16_CL1.select<TILE_N, 1>(0)  = result1ref.select<1, 1, TILE_N, 1>(i, 0);
		cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
	}

#endif // !defined(EMPTY)
}
