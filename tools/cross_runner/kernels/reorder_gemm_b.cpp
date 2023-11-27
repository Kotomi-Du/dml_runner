#include <cm/cm.h>
#include <cm/cmtl.h>

#define HALF half
#define FLOAT float
#define DIM_X 0
#define DIM_Y 1
#define DIM_Z 2

#define SIZE_OF_HF16_BYTE 2

#define INPUT_TRANSPOSED 0

extern "C" _GENX_MAIN_ void gemm_reorder(
	SurfaceIndex surface_input [[type("buffer_t")]],
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

	const uint32_t thread_id_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
	const uint32_t thread_id_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
	const uint32_t thread_id_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);
	
	const uint32_t base_offset =  thread_id_0 * TILE_K * SIZE_N * SIZE_OF_HF16_BYTE;
	const uint32_t base_offset_output =  (thread_id_0 * TILE_K * SIZE_N + thread_id_1 * TILE_N) * SIZE_OF_HF16_BYTE;
    


#if !INPUT_TRANSPOSED
	const uint32_t step_base_offset = base_offset; // + (step / TILE_K) * SIZE_N * TILE_K * SIZE_OF_HF16_BYTE;

	// TILE_N = SIZE_N
	matrix<HALF, TILE_K/2, TILE_N * 2> reordered_matrix = 0.0;  

	//cache elements in matrix B 
	#pragma unroll
	for(int row = 0; row < TILE_K/2; row++)
	{
		const uint32_t rowX2 = row * 2;
		const uint32_t read_offset = step_base_offset + (rowX2 * SIZE_N)* SIZE_OF_HF16_BYTE;
		reordered_matrix.select<1,1,TILE_N,2>(row,0).format<U32>() = cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, read_offset);  
		reordered_matrix.select<1,1,TILE_N,2>(row,1).format<U32>()= cm_load<U32, TILE_N/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_input, read_offset + SIZE_N* SIZE_OF_HF16_BYTE);  
	}

	if (thread_id_0 == 0 && thread_id_1 == 0)
	{
		for (int i = 0; i < 8; i++)
		{
			printf(" row%d", i);
			for (int j = 0; j < TILE_N*2;j++)
			{ 
				printf(" %f",reordered_matrix(i,j));
				// printf(" %f",readB(i *16+j));
			}
			printf("\n");\
		}
	}

	vector<HALF, TILE_N*2> result_hf16_CL1;
	#pragma unroll
	for(int i = 0; i < TILE_K/2; i++)
	{
		const unsigned write_index = base_offset_output + i * SIZE_N * 2* SIZE_OF_HF16_BYTE;	
		
		result_hf16_CL1.select<TILE_N*2, 1>(0)  = reordered_matrix.select<1,1,TILE_N*2,1>(i,0);
		cm_store<U32, TILE_N, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface_output, write_index, result_hf16_CL1.format<U32>());
	}	
#endif

}
