#include <cm/cm.h>
#include <cm/cmtl.h>

#define MSG_SIMD_SIZE 16
#define ROWS_PER_TILE 8
#define HALF half
#define FLOAT float
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2

#define SG_TILE_NUM_ROWS 8
#define SIZE_OF_HF16_BYTE 2

#define INPUT_B_OFFSET ((SIZE_N * TILE_K)* sizeof(HALF))
static const int32_t init_linear_offsets[] = {  0  ,
											    1  , 
											    2  ,
											    3  ,
											    4  ,
											    5  * INPUT_B_OFFSET,
											    5  * INPUT_B_OFFSET,
											    7  * INPUT_B_OFFSET,
												8  * INPUT_B_OFFSET, 
											 };

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

	const uint32_t thread_id_0 = tidX * gidX + tidX;
	const uint32_t thread_id_1 = tidY * gidY + tidY;
	const unsigned base_offset_a =  thread_id_0 * TILE_M * SIZE_K * SIZE_OF_HF16_BYTE;
	const unsigned base_offset_b =  thread_id_1 * TILE_N * SIZE_OF_HF16_BYTE;
	const unsigned base_offset_output =  thread_id_0 * TILE_M * TILE_N * SIZE_OF_HF16_BYTE;
    
	vector<HALF, 128> readA1 = 0.0; 	// M=0..7,  K=0..15		//A matrix format: [K/16][M][16k] : A tile: 8Mx16K
	vector<HALF, 128> readB1 = 0.0; 	// N=0..7,  K=0..15		//B matrix format: [K/16][N/8][8K][8N][2K]	//B tile: 40Nx16K
	vector<HALF, 128> readB2 = 0.0; 	// N=8..15, K=0..15
	vector<HALF, 128> readB3 = 0.0; 	// N=16..23,K=0..15
	vector<HALF, 128> readB4 = 0.0; 	// N=24..31,K=0..15

	//referrence variables	
	matrix_ref<HALF, 8, 16> readA1_m = readA1.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB1_m = readB1.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB2_m = readB2.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB3_m = readB3.format<HALF, 8, 16>();
	matrix_ref<HALF, 8, 16> readB4_m = readB4.format<HALF, 8, 16>();

	//init the accumulators
	matrix<FLOAT, 8, 8> result11 = 0.0; 
	matrix<FLOAT, 8, 8> result12 = 0.0; 
	matrix<FLOAT, 8, 8> result13 = 0.0; 
	matrix<FLOAT, 8, 8> result14 = 0.0; 
	
	matrix_ref<FLOAT, 8, 8> result11ref = result11;
	matrix_ref<FLOAT, 8, 8> result12ref = result12;
	matrix_ref<FLOAT, 8, 8> result13ref = result13;
	matrix_ref<FLOAT, 8, 8> result14ref = result14;

	vector<ushort, MSG_SIMD_SIZE> predicate;

	for( int step = 0; step < SIZE_K; step += 16)
	{
		const unsigned step_base_offset_a = base_offset_a + step * SIZE_OF_HF16_BYTE;
		const unsigned step_base_offset_b = base_offset_b + step * SIZE_N * TILE_K * SIZE_OF_HF16_BYTE;
		
		readA1.select_all() = 0.0;
		readB1.select_all() = 0.0;
		readB2.select_all() = 0.0;
		readB3.select_all() = 0.0;
		readB4.select_all() = 0.0;

		#pragma unroll
		for(int row = 0; row < 8; row++)
		{
			const unsigned row_offset_in_bytes = row * ROWS_PER_TILE * SIZE_OF_HF16_BYTE;
			const unsigned read_offset_a = step_base_offset_a + (row * SIZE_K) * SIZE_OF_HF16_BYTE;
			// Read from inputs surfaces 16K X 8M
			readA1.select<16,1>(row_offset_in_bytes).format<U32>() = cm_load<U32, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_a, read_offset_a);
			
			const unsigned rowX2 = row * 2;
			vector<uint32_t, 16> base_b_offsets(init_linear_offsets);
			const unsigned read_offset_b = step_base_offset_b + (row * SIZE_N)* SIZE_OF_HF16_BYTE;
			base_b_offsets += read_offset_b;
			vector<uint32_t, 16>  input_surface_CL1 = base_b_offsets;
			vector<uint32_t, 16>  input_surface_CL2 = base_b_offsets + TILE_N;
			vector<uint32_t, 16>  input_surface_CL3 = base_b_offsets + TILE_N * 2;
			vector<uint32_t, 16>  input_surface_CL4 = base_b_offsets + TILE_N * 3;
			
			// Read 16K x 32N and Rearrange Matrix B for DPAS compatibility
			readB1_m.select<8,1,1,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b);
			readB1_m.select<8,1,1,1>(0, rowX2+1).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + (rowX2+1)*SIZE_N);
			readB2_m.select<8,1,1,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + TILE_N);
			readB2_m.select<8,1,1,1>(0, rowX2+1).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + TILE_N + (rowX2+1)*SIZE_N);
			readB3_m.select<8,1,1,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + TILE_N * 2);
			readB3_m.select<8,1,1,1>(0, rowX2+1).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + TILE_N * 2 + (rowX2+1)*SIZE_N);
			readB4_m.select<8,1,1,1>(0, rowX2).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + TILE_N * 3);
			readB4_m.select<8,1,1,1>(0, rowX2+1).format<U32>() = cm_load<U32, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, read_offset_b + TILE_N * 3 +  (rowX2+1)*SIZE_N);
			// readB1_m.select<16,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, VectorSize::N1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_surface_CL1);
			// readB2_m.select<16,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, VectorSize::N1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_surface_CL2);
			// readB3_m.select<16,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, VectorSize::N1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_surface_CL3);
			// readB4_m.select<16,1,2,1>(0, rowX2).format<U32>() = cm_load<U32, VectorSize::N1, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input_b, input_surface_CL4);
		}
		myDPAS8(readA1_m, readB1_m, result11ref);
		myDPAS8(readA1_m, readB2_m, result12ref);
		myDPAS8(readA1_m, readB3_m, result13ref);
		myDPAS8(readA1_m, readB4_m, result14ref);

	}
	vector<HALF, 32> result_hf16_CL1 = 0.0;
	result11 *= HALF(ALPHA);
	result12 *= HALF(ALPHA);
	result13 *= HALF(ALPHA);
	result14 *= HALF(ALPHA);
	
	#pragma unroll
	for(int i = 0; i < SG_TILE_NUM_ROWS; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * SIZE_OF_HF16_BYTE;
		
		result_hf16_CL1.select<8, 1>(0)  = result11ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(8)  = result12ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(16) = result13ref.select<1, 1, 8, 1>(i, 0);
		result_hf16_CL1.select<8, 1>(24) = result14ref.select<1, 1, 8, 1>(i, 0);
		
		cm_store<U32, 16, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
	}

#endif // !defined(EMPTY)
}
