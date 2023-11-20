#include <cm/cm.h>
#include <cm/cmtl.h>

#define HALF half
#define FLOAT float
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2

#define SIZE_OF_HF16_BYTE 2

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
    // TILE SIZE of 8M x 32N x 16K [tile_height x tile_width x step size]
	// use byte to calcute offset
	uint gidX = cm_group_id(DIM_X);
	uint gidY = cm_group_id(DIM_Y);
	uint gidZ = cm_group_id(DIM_Z);
    uint tidX = cm_local_id(DIM_X);
    uint tidY = cm_local_id(DIM_Y);
	uint tidZ = cm_local_id(DIM_Z);

	const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const unsigned base_offset_a =  thread_id_0 * TILE_M * SIZE_K * SIZE_OF_HF16_BYTE;
	const unsigned base_offset_b =  thread_id_1 * TILE_N * SIZE_OF_HF16_BYTE;
	const unsigned base_offset_output =  (thread_id_0 * TILE_M * SIZE_N + thread_id_1 * TILE_N) * SIZE_OF_HF16_BYTE;
    
	vector<HALF, 128> readA1 = 0.0; 	// M=0..7,  K=0..15		//A matrix format: [K/16][M][16k] : A tile: 8Mx16K
	vector<HALF, 128> readA2 = 0.0; 	// M=0..7,  K=0..15		//A matrix format: [K/16][M][16k] : A tile: 8Mx16K
	vector<HALF, 128> readB = 0.0; 	// N=0..7,  K=0..15		//B matrix format: [K/16][N/8][8K][8N][2K]	//B tile: 40Nx16K

	//referrence variables	
	matrix_ref<HALF, 8, 16> readA1_m = readA1.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readA2_m = readA2.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB_m = readB.format<HALF, 8, 16>();

	//init the accumulators
	matrix<FLOAT, 8, TILE_N> result1 = 0.0;  
	matrix<FLOAT, 8, TILE_N> result2 = 0.0; 
	
	matrix_ref<FLOAT, 8, TILE_N> result1ref = result1;
	matrix_ref<FLOAT, 8, TILE_N> result2ref = result2;


	for( int step = 0; step < SIZE_K; step += TILE_K)
	{
		const unsigned step_base_offset_a = base_offset_a + step * SIZE_OF_HF16_BYTE;
		const unsigned step_base_offset_b = base_offset_b + (step / TILE_K) * SIZE_N * TILE_K * SIZE_OF_HF16_BYTE;
		
		matrix<HALF, 8, TILE_N> rowX2_0 = 0.0;  
		matrix<HALF, 8, TILE_N> rowX2_1 = 0.0;
		#pragma unroll
		for(int row = 0; row < 8; row++)
		{
			const unsigned vector_offset_a = row * TILE_K;
			const unsigned read_offset_a = step_base_offset_a + (row * SIZE_K) * SIZE_OF_HF16_BYTE;
			// Read from inputs surfaces row M x 16K
			readA1.select<16,1>(vector_offset_a).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, read_offset_a);
			readA2.select<16,1>(vector_offset_a).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, read_offset_a + SIZE_K * 8 * SIZE_OF_HF16_BYTE);		

			const unsigned rowX2 = row * 2;
			const unsigned read_offset_b = step_base_offset_b + (rowX2 * SIZE_N)* SIZE_OF_HF16_BYTE;
			rowX2_0.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);  
			rowX2_1.select<1,1,TILE_N,1>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + SIZE_N* SIZE_OF_HF16_BYTE);  

		}	
		
		for(int n = 0; n < TILE_N/8; n++)  
		{
			#pragma unroll
			for (int row = 0; row < 8; row++)
			{
				readB_m.select<1,1,8,2>(row, 0)= rowX2_0.select<1,1,8,1>(row,8*n);
				readB_m.select<1,1,8,2>(row, 1)= rowX2_1.select<1,1,8,1>(row,8*n);
			}
			
			myDPAS8(readA1_m, readB_m, result1ref.select<8,1,8,1>(0,n*8));  
			myDPAS8(readA2_m, readB_m, result2ref.select<8,1,8,1>(0,n*8)); 
		}

		// if (gidX == 0 && gidY == 0 && gidZ == 0)
		// {
		// 	for (int i = 0; i < 8; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 8; j < 8*2;j++)
        // 		{ 
        // 		    printf(" %f", result1ref(i , j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }
		// if (gidX == 0 && gidY == 0 && gidZ == 0)
		// {
		// 	for (int i = 0; i < 8; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < 16;j++)
        // 		{ 
        // 		    printf(" %f",readA2(i *16+j));
		// 			//printf(" %f",readB1(i *16+j));
        // 		}
        // 		printf("\n");\
    	// 	}
		// }

	}
	vector<HALF, TILE_N> result_hf16_CL1 = 0.0;
	result1 *= HALF(ALPHA);
	vector<HALF, TILE_N> result_hf16_CL2 = 0.0;
	result2 *= HALF(ALPHA);
	
	#pragma unroll
	for(int i = 0; i < TILE_M/2; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_HF16_BYTE;
		
		result_hf16_CL1.select<TILE_N, 1>(0)  = result1ref.select<1, 1, TILE_N, 1>(i, 0);
		result_hf16_CL2.select<TILE_N, 1>(0)  = result2ref.select<1, 1, TILE_N, 1>(i, 0);

		
		cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
		cm_store<U32, TILE_N/2, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index + 8*SIZE_N*SIZE_OF_HF16_BYTE, result_hf16_CL2.format<U32>());
	}

#endif // !defined(EMPTY)
}
