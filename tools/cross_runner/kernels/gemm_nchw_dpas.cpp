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
	vector<HALF, 128> readB1 = 0.0; 	// N=0..7,  K=0..15		//B matrix format: [K/16][N/8][8K][8N][2K]	//B tile: 40Nx16K
	vector<HALF, 128> readB2 = 0.0; 	// N=8..15, K=0..15
	vector<HALF, 128> readB3 = 0.0; 	// N=16..23,K=0..15
	vector<HALF, 128> readB4 = 0.0; 	// N=24..31,K=0..15
	vector<HALF, 128> readB5 = 0.0; 	// N=0..7,  K=0..15		//B matrix format: [K/16][N/8][8K][8N][2K]	//B tile: 40Nx16K
	vector<HALF, 128> readB6 = 0.0; 	// N=8..15, K=0..15
	vector<HALF, 128> readB7 = 0.0; 	// N=16..23,K=0..15
	vector<HALF, 128> readB8 = 0.0; 	// N=24..31,K=0..15
	// vector<HALF, 128> readB9 = 0.0; 	// N=16..23,K=0..15
	// vector<HALF, 128> readB10 = 0.0; 	// N=24..31,K=0..15

	//referrence variables	
	matrix_ref<HALF, 8, 16> readA1_m = readA1.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readA2_m = readA2.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB1_m = readB1.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB2_m = readB2.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB3_m = readB3.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB4_m = readB4.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB5_m = readB5.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB6_m = readB6.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB7_m = readB7.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB8_m = readB8.format<HALF, 8, 16>();
	// matrix_ref<HALF, 8, 16> readB9_m = readB9.format<HALF, 8, 16>();
	// matrix_ref<HALF, 8, 16> readB10_m = readB10.format<HALF, 8, 16>();

	//init the accumulators
	matrix<FLOAT, 8, 8> result11 = 0.0; 
	matrix<FLOAT, 8, 8> result12 = 0.0; 
	matrix<FLOAT, 8, 8> result13 = 0.0; 
	matrix<FLOAT, 8, 8> result14 = 0.0; 
	matrix<FLOAT, 8, 8> result15 = 0.0; 
	matrix<FLOAT, 8, 8> result16 = 0.0; 
	matrix<FLOAT, 8, 8> result17 = 0.0; 
	matrix<FLOAT, 8, 8> result18 = 0.0; 
	// matrix<FLOAT, 8, 8> result19 = 0.0; 
	// matrix<FLOAT, 8, 8> result110 = 0.0; 

	matrix<FLOAT, 8, 8> result21 = 0.0; 
	matrix<FLOAT, 8, 8> result22 = 0.0; 
	matrix<FLOAT, 8, 8> result23 = 0.0; 
	matrix<FLOAT, 8, 8> result24 = 0.0; 
	matrix<FLOAT, 8, 8> result25 = 0.0; 
	matrix<FLOAT, 8, 8> result26 = 0.0; 
	matrix<FLOAT, 8, 8> result27 = 0.0; 
	matrix<FLOAT, 8, 8> result28 = 0.0; 
	// matrix<FLOAT, 8, 8> result29 = 0.0; 
	// matrix<FLOAT, 8, 8> result210 = 0.0; 
	
	matrix_ref<FLOAT, 8, 8> result11ref = result11;
	matrix_ref<FLOAT, 8, 8> result12ref = result12;
	matrix_ref<FLOAT, 8, 8> result13ref = result13;
	matrix_ref<FLOAT, 8, 8> result14ref = result14;
	matrix_ref<FLOAT, 8, 8> result15ref = result15;
	matrix_ref<FLOAT, 8, 8> result16ref = result16;
	matrix_ref<FLOAT, 8, 8> result17ref = result17;
	matrix_ref<FLOAT, 8, 8> result18ref = result18;
	// matrix_ref<FLOAT, 8, 8> result19ref = result19;
	// matrix_ref<FLOAT, 8, 8> result110ref = result110;
	matrix_ref<FLOAT, 8, 8> result21ref = result21;
	matrix_ref<FLOAT, 8, 8> result22ref = result22;
	matrix_ref<FLOAT, 8, 8> result23ref = result23;
	matrix_ref<FLOAT, 8, 8> result24ref = result24;
	matrix_ref<FLOAT, 8, 8> result25ref = result25;
	matrix_ref<FLOAT, 8, 8> result26ref = result26;
	matrix_ref<FLOAT, 8, 8> result27ref = result27;
	matrix_ref<FLOAT, 8, 8> result28ref = result28;
	// matrix_ref<FLOAT, 8, 8> result29ref = result29;
	// matrix_ref<FLOAT, 8, 8> result210ref = result210;

	for( int step = 0; step < SIZE_K; step += TILE_K)
	{
		const unsigned step_base_offset_a = base_offset_a + step * SIZE_OF_HF16_BYTE;
		const unsigned step_base_offset_b = base_offset_b + (step / TILE_K) * SIZE_N * TILE_K * SIZE_OF_HF16_BYTE;
		
		readA1.select_all() = 0.0;
		readA2.select_all() = 0.0;
		readB1.select_all() = 0.0;
		readB2.select_all() = 0.0;
		readB3.select_all() = 0.0;
		readB4.select_all() = 0.0;
		readB5.select_all() = 0.0;
		readB6.select_all() = 0.0;
		readB7.select_all() = 0.0;
		readB8.select_all() = 0.0;
		// readB9.select_all() = 0.0;
		// readB10.select_all() = 0.0;

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
			// Read 2K x 32N and Rearrange Matrix B for DPAS compatibility
			// less load and store
			vector<uint32_t, 32> rowX2_0_packed =  cm_load<U32, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);  
			vector<uint32_t, 32> rowX2_1_packed =  cm_load<U32, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b,  read_offset_b + SIZE_N* SIZE_OF_HF16_BYTE); 
			vector_ref<half, 64> rowX2_0 = rowX2_0_packed.format<half>();  
			vector_ref<half, 64> rowX2_1 = rowX2_1_packed.format<half>();  
			readB1_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(0);
			readB2_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(8);
			readB3_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(16);
			readB4_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(24);
			readB5_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(32);
			readB6_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(40);
			readB7_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(48);
			readB8_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(56);
			// readB9_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(64);
			// readB10_m.select<1,1,8,2>(row, 0)= rowX2_0.select<8,1>(72);

			readB1_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(0);
			readB2_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(8);
			readB3_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(16);
			readB4_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(24);
			readB5_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(32);
			readB6_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(40);
			readB7_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(48);
			readB8_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(56);
			// readB9_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(64);
			// readB10_m.select<1,1,8,2>(row, 1)= rowX2_1.select<8,1>(72);
		}
		
		myDPAS8(readA1_m, readB1_m, result11ref);  
		myDPAS8(readA1_m, readB2_m, result12ref);
		myDPAS8(readA1_m, readB3_m, result13ref);
		myDPAS8(readA1_m, readB4_m, result14ref);
		myDPAS8(readA1_m, readB5_m, result15ref);  
		myDPAS8(readA1_m, readB6_m, result16ref);
		myDPAS8(readA1_m, readB7_m, result17ref);
		myDPAS8(readA1_m, readB8_m, result18ref);
		// myDPAS8(readA1_m, readB7_m, result19ref);
		// myDPAS8(readA1_m, readB8_m, result110ref);

		myDPAS8(readA2_m, readB1_m, result21ref);  
		myDPAS8(readA2_m, readB2_m, result22ref);
		myDPAS8(readA2_m, readB3_m, result23ref);
		myDPAS8(readA2_m, readB4_m, result24ref);
		myDPAS8(readA2_m, readB5_m, result25ref);  
		myDPAS8(readA2_m, readB6_m, result26ref);
		myDPAS8(readA2_m, readB7_m, result27ref);
		myDPAS8(readA2_m, readB8_m, result28ref);
		// myDPAS8(readA2_m, readB7_m, result29ref);
		// myDPAS8(readA2_m, readB8_m, result210ref);
		// if (gidX == 0 && gidY == 0 && gidZ == 0)
		// {
		// 	for (int i = 0; i < 8; i++)
    	// 	{
        // 		  printf(" row%d", i);
       	// 		 for (int j = 0; j < 8;j++)
        // 		{ 
        // 		    printf(" %f", result21ref(i , j));
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
	vector<HALF, 64> result_hf16_CL1 = 0.0;
	result11 *= HALF(ALPHA);
	result12 *= HALF(ALPHA);
	result13 *= HALF(ALPHA);
	result14 *= HALF(ALPHA);
	result15 *= HALF(ALPHA);
	result16 *= HALF(ALPHA);
	result17 *= HALF(ALPHA);
	result18 *= HALF(ALPHA);
	// result19 *= HALF(ALPHA);
	// result110 *= HALF(ALPHA);
	vector<HALF, 64> result_hf16_CL2 = 0.0;
	result21 *= HALF(ALPHA);
	result22 *= HALF(ALPHA);
	result23 *= HALF(ALPHA);
	result24 *= HALF(ALPHA);
	result25 *= HALF(ALPHA);
	result26 *= HALF(ALPHA);
	result27 *= HALF(ALPHA);
	result28 *= HALF(ALPHA);
	// result29 *= HALF(ALPHA);
	// result210 *= HALF(ALPHA);
	
	#pragma unroll
	for(int i = 0; i < TILE_M/2; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_HF16_BYTE;
		
		result_hf16_CL1.select<8, 1>(0)  = result11ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(8)  = result12ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(16) = result13ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(24) = result14ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(32) = result15ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(40) = result16ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(48) = result17ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(56) = result18ref.select<1, 1, 8, 1>(i, 0);

		result_hf16_CL2.select<8, 1>(0)  = result21ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(8)  = result22ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(16) = result23ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(24) = result24ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(32) = result25ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(40) = result26ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(48) = result27ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL2.select<8, 1>(56) = result28ref.select<1, 1, 8, 1>(i, 0);

		
		cm_store<U32, 32, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
		cm_store<U32, 32, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index + 8*SIZE_N*SIZE_OF_HF16_BYTE, result_hf16_CL2.format<U32>());
	}

#endif // !defined(EMPTY)
}
