/*========================== begin_copyright_notice ============================

INTEL CONFIDENTIAL

Copyright (C) 2023 Intel Corporation

This software and the related documents are Intel copyrighted materials,
and your use of them is governed by the express license under which they were
provided to you ("License"). Unless the License provides otherwise,
you may not use, modify, copy, publish, distribute, disclose or transmit this
software or the related documents without Intel's prior written permission.

This software and the related documents are provided as is, with no express or
implied warranties, other than those that are expressly stated in the License.

============================= end_copyright_notice ===========================*/

#include <cm/cm.h>
#include <cm/cmtl.h>

#define DEBUG_MODE 0
#if !CM_HAS_DPAS
#error [Error_device_no_dpas] Kernel designed to use dpas. Current device does not support dpas.
#endif

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#if(CM_GENX >= 1280)
#error [Error_device_not_supported] Kernel is not designed for Xe2+ architecutre.
#endif

#if BLOCK_W > 8
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_w in range: <1; 7>;
#endif

#if BLOCK_OC != 8 && BLOCK_OC != 16 && BLOCK_OC != 32 && BLOCK_OC != 40 && BLOCK_OC != 64 && BLOCK_OC != 80
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_oc which is equal to 8 or 16 or 32 or 40 or 64 or 80;
#endif

#define DPAS_DEPTH 8 
#if(CM_GENX >= 1280)
#define EXEC_SIZE 16
#else
#define EXEC_SIZE 8
#endif

#define BLOCK_H 1
#define WIDTH_LEFTOVER (OUTPUT_WIDTH % BLOCK_W)
#define HAS_LEFTOVER (WIDTH_LEFTOVER != 0)
#define LEFTOVER_COVERS_FULL_WIDTH (OUTPUT_WIDTH == WIDTH_LEFTOVER)

#define DT_OUT half
#define DT_IN half
#define DT_IN_SIZE 2 
#define DT_WEIGHTS half
// accu on DG2 have to be float for half dt inputs
#define DT_ACCU float 

#define DWORD_SIZE 4
#define INPUT_WIDTH_ALIGNED_TO_DWORD ((INPUT_WIDTH * DT_IN_SIZE) % DWORD_SIZE == 0)

#define DPAS_INPUT_CHANNELS (DPAS_DEPTH * sizeof(DT_IN))
#define DPAS_OUTPUT_CHANNELS EXEC_SIZE
#define DPAS_RC BLOCK_W

#define CONV_LOOP_COUNT ((INPUT_CHANNELS/DPAS_INPUT_CHANNELS) / SLICE_IC)

#define WEIGHTS_REG_SIZE (DPAS_INPUT_CHANNELS * DPAS_OUTPUT_CHANNELS)
#define WEIGHTS_IC_OFSET sizeof(DT_WEIGHTS)
#define WEIGHTS_OC_OFSET (INPUT_CHANNELS * sizeof(DT_WEIGHTS))

#define INPUT_NCHW_PLANE_SIZE (INPUT_WIDTH * INPUT_HEIGHT * sizeof(DT_IN))
#define OUTPUT_NCHW_PLANE_SIZE (OUTPUT_WIDTH * OUTPUT_HEIGHT * sizeof(DT_OUT))

static const uint32_t init_linear_offsets[] = { 0, 2, 4, 6, 8, 10, 12, 14 };
static const uint32_t output_init_offsets[] = {
                                            0 * OUTPUT_NCHW_PLANE_SIZE, 1 * OUTPUT_NCHW_PLANE_SIZE,
                                            2 * OUTPUT_NCHW_PLANE_SIZE, 3 * OUTPUT_NCHW_PLANE_SIZE,
                                            4 * OUTPUT_NCHW_PLANE_SIZE, 5 * OUTPUT_NCHW_PLANE_SIZE,
                                            6 * OUTPUT_NCHW_PLANE_SIZE, 7 * OUTPUT_NCHW_PLANE_SIZE,
                                            };

static const uint32_t weights_init_offsets[] = {
                                                0 * WEIGHTS_OC_OFSET, 0 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                1 * WEIGHTS_OC_OFSET, 1 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                2 * WEIGHTS_OC_OFSET, 2 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                3 * WEIGHTS_OC_OFSET, 3 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                4 * WEIGHTS_OC_OFSET, 4 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                5 * WEIGHTS_OC_OFSET, 5 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                6 * WEIGHTS_OC_OFSET, 6 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                7 * WEIGHTS_OC_OFSET, 7 * WEIGHTS_OC_OFSET + WEIGHTS_IC_OFSET,
                                                };

template<uint32_t LOAD_W>
_GENX_ inline vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> load_input_nchw_and_reorder_to_wc16(SurfaceIndex surface [[type("buffer_t")]], uint byte_offset)
{
    const uint32_t LOAD_W_WIDTH = LOAD_W * STRIDE_W;
    const uint32_t LOAD_W_BYTES_WIDTH = LOAD_W_WIDTH * sizeof(DT_IN);
    const uint32_t LOAD_W_DWORDS = LOAD_W_BYTES_WIDTH / sizeof(uint32_t);
    
    vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> data_out;
#if INPUT_WIDTH_ALIGNED_TO_DWORD && BLOCK_W == 8
    #pragma unroll
    for(int i = 0; i < DPAS_INPUT_CHANNELS; i++)
    {
        vector<uint32_t, LOAD_W_DWORDS> load_chunk = cm_load<uint32_t, LOAD_W_DWORDS, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
        vector<half, LOAD_W_WIDTH> load_chunk_typed = load_chunk.format<half>();  
        data_out.select<BLOCK_W, DPAS_INPUT_CHANNELS>(i) = load_chunk_typed.select<BLOCK_W, STRIDE_W>();
        byte_offset += INPUT_NCHW_PLANE_SIZE;
    }  
#else
    // non transposed scattered reads
    vector<uint32_t, 8> offsets(init_linear_offsets);
    offsets += byte_offset;
    #pragma unroll
    for(int i = 0; i < DPAS_INPUT_CHANNELS; i++)
    {
        vector<half, 8> w_grf_chunk = cm_load<half, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets);
        data_out.select<8, DPAS_INPUT_CHANNELS>(i) = w_grf_chunk;
        offsets += INPUT_NCHW_PLANE_SIZE;
    }
#endif
    return data_out;
}

_GENX_ inline vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> load_filter_nchw_data(SurfaceIndex surface [[type("buffer_t")]], uint32_t byte_offset)
{
    //static_assert(KERNEL_SIZE == 1, "Weights loading in this kernel is implemented only for 1x1 weights size");
    const uint32_t PACKED_ELEMENT = sizeof(uint32_t)/ sizeof(DT_WEIGHTS);
    const uint32_t INPUT_CHANNELS_CHUNKS = DPAS_INPUT_CHANNELS / PACKED_ELEMENT;
    const uint32_t LOAD_SIZE = PACKED_ELEMENT * DPAS_OUTPUT_CHANNELS;
    vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> data_out;
#if !WEIGHTS_IN_OPTIMAL_FORMAT
    vector<DT_WEIGHTS, DPAS_INPUT_CHANNELS> data_load;
    vector_ref<uint32_t, DPAS_INPUT_CHANNELS / 2> data_load_view = data_load.format<uint32_t>();
    #pragma unroll
    for(int i = 0; i < DPAS_OUTPUT_CHANNELS; i++)
    {
        data_load_view = cm_load<uint32_t, DPAS_INPUT_CHANNELS/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);  
        byte_offset += WEIGHTS_OC_OFSET;
        #pragma unroll
        for(int j = 0; j < DPAS_OUTPUT_CHANNELS; j++)
        {
            data_out.select<2, 1>(16 * j + i * 2) = data_load.select<2, 1>(j * 2); 
        }

    }
#else
    vector_ref<uint32_t, 64> data_load_view =data_out.format<uint32_t>();
    data_load_view = cm_load<uint32_t, 64, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);  
#endif
    return data_out;
}

_GENX_ inline vector<DT_OUT, BLOCK_OC> load_bias(SurfaceIndex surface [[type("buffer_t")]], uint32_t byte_offset)
{
	vector<DT_OUT, BLOCK_OC> data_out;
#if BLOCK_OC == 40
	data_out.select<32,1>(0).format<uint32_t>() = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
	data_out.select<8,1>(32).format<uint32_t>() = cm_load<uint32_t, 4, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset + 64);
#endif
#if BLOCK_OC == 80
	data_out.select<64,1>(0).format<uint32_t>() = cm_load<uint32_t, 32, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
	data_out.select<16,1>(64).format<uint32_t>() = cm_load<uint32_t, 8, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset + 128);
#endif
#if BLOCK_OC != 40 && BLOCK_OC != 80
	data_out.format<uint32_t>() = cm_load<uint32_t, BLOCK_OC/2, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, byte_offset);
#endif
	return data_out;
}

template<uint32_t STORE_W>
_GENX_ inline void store_output_wc8_as_nchw(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, BLOCK_W * DPAS_OUTPUT_CHANNELS> grf_chunk, uint32_t byte_offset, uint32_t w_chunk_id)
{    

#if HAS_LEFTOVER
    if(w_chunk_id == ((details::roundUpNextMultiple(OUTPUT_WIDTH, BLOCK_W)/BLOCK_W) -1))
    {
        // non transposed scattered writes
        vector<uint32_t, DPAS_OUTPUT_CHANNELS> offsets_leftovers(output_init_offsets);
        offsets_leftovers += byte_offset;
        #pragma unroll
        for(int i = 0; i < WIDTH_LEFTOVER; i++)
        {
            // pick data to store
            vector_ref<DT_OUT, DPAS_OUTPUT_CHANNELS> grf_chunk_store = grf_chunk.select<DPAS_OUTPUT_CHANNELS, 1>(i * DPAS_OUTPUT_CHANNELS);
            // store with non-transposed msg
            cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets_leftovers, grf_chunk_store);
            offsets_leftovers += sizeof(DT_OUT);  // move by one element
        }  
    }
    else
    {
#endif
#if !LEFTOVER_COVERS_FULL_WIDTH
        vector<uint32_t, STORE_W> offsets(init_linear_offsets);
        offsets += byte_offset;
        // block stores could be used here, but I havent seen any advtange of using it yet. Leaving this code for  
        #pragma unroll
        for(int i = 0; i < DPAS_OUTPUT_CHANNELS; i++)
        {
            // pick data to store
            vector<DT_OUT, STORE_W> grf_chunk_store = grf_chunk.select<STORE_W, DPAS_OUTPUT_CHANNELS>(i);                  
            cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store);
            offsets += OUTPUT_NCHW_PLANE_SIZE;
        }
#endif // !LEFTOVER_COVERS_FULL_WIDTH
#if HAS_LEFTOVER
    }
#endif
}

extern "C" _GENX_MAIN_ void convolution_nchw_1x1(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
#if USE_BIAS
	SurfaceIndex surface_bias [[type("buffer_t")]],
#endif
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    const uint32_t thg_0 = (cm_group_id(0) * cm_local_size(0) + cm_local_id(0));
    const uint w_chunk_id = thg_0 / SLICE_IC;
    const uint slice_ic_id = thg_0 % SLICE_IC;
    const uint h_chunk_id = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint thread_id_2 = (cm_group_id(2) * cm_local_size(2) + cm_local_id(2));
    
    const uint THREADS_FOR_OC = (OUTPUT_CHANNELS / BLOCK_OC);
    const uint batch_id = (thread_id_2 / THREADS_FOR_OC);
    const uint oc_chunk_id = (thread_id_2 % THREADS_FOR_OC) * (BLOCK_OC / DPAS_OUTPUT_CHANNELS);
    
    const uint32_t input_row_offset_size = INPUT_WIDTH;
    const uint32_t input_dpas_ic_offset_size = INPUT_HEIGHT * DPAS_INPUT_CHANNELS * input_row_offset_size;
    
    const uint32_t input_batch_offset = batch_id * INPUT_WIDTH * INPUT_HEIGHT * INPUT_CHANNELS;
    const uint32_t input_w_chunk_offset = w_chunk_id * BLOCK_W * STRIDE_W;
    const uint32_t input_h_chunk_offset = h_chunk_id * BLOCK_H * STRIDE_H * input_row_offset_size;
    const uint32_t input_slice_ic_chunk_offset = slice_ic_id * CONV_LOOP_COUNT * input_dpas_ic_offset_size;
    uint32_t input_offset = (input_batch_offset + input_slice_ic_chunk_offset + input_h_chunk_offset + input_w_chunk_offset) * sizeof(DT_IN);
        
        
#if WEIGHTS_IN_OPTIMAL_FORMAT
    const uint32_t weights_oc_chunk_offset = EXEC_SIZE * DPAS_INPUT_CHANNELS * sizeof(DT_WEIGHTS);
    const uint32_t weights_ic_offset_size = OUTPUT_CHANNELS * DPAS_INPUT_CHANNELS * sizeof(DT_WEIGHTS);
#else
    const uint32_t weights_oc_chunk_offset = DPAS_OUTPUT_CHANNELS * INPUT_CHANNELS * sizeof(DT_WEIGHTS);
    const uint32_t weights_ic_offset_size = DPAS_INPUT_CHANNELS * sizeof(DT_WEIGHTS);
#endif

    uint32_t weights_offset_0 = oc_chunk_id * weights_oc_chunk_offset + (slice_ic_id * CONV_LOOP_COUNT * weights_ic_offset_size);
    uint32_t weights_offset_1 = weights_offset_0 + weights_oc_chunk_offset;
    uint32_t weights_offset_2 = weights_offset_1 + weights_oc_chunk_offset;
    uint32_t weights_offset_3 = weights_offset_2 + weights_oc_chunk_offset;
	uint32_t weights_offset_4 = weights_offset_3 + weights_oc_chunk_offset;
    uint32_t weights_offset_5 = weights_offset_4 + weights_oc_chunk_offset;
	uint32_t weights_offset_6 = weights_offset_5 + weights_oc_chunk_offset;
    uint32_t weights_offset_7 = weights_offset_6 + weights_oc_chunk_offset;
    uint32_t weights_offset_8 = weights_offset_7 + weights_oc_chunk_offset;
    uint32_t weights_offset_9 = weights_offset_8 + weights_oc_chunk_offset;
	
    const uint ACCU_REG_SIZE = BLOCK_W * DPAS_OUTPUT_CHANNELS;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_0 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_1 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_2 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_3 = 0;
	vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_4 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_5 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_6 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_7 = 0;
	vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_8 = 0;
    vector<DT_ACCU, ACCU_REG_SIZE> accu_row_0_oc_9 = 0;
	
    // todo debug performance with pragma unroll
    //#pragma unroll
    for(int i = 0; i < CONV_LOOP_COUNT; i++)
    {
        vector<DT_IN, BLOCK_W * DPAS_INPUT_CHANNELS> input_row_0 = load_input_nchw_and_reorder_to_wc16<BLOCK_W>(surface_input, input_offset);
		
        vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_0 = load_filter_nchw_data(surface_weights, weights_offset_0);

#if DEBUG_MODE
        if (thg_0 == 0 && h_chunk_id == 0 && thread_id_2==0 )
         {
            printf("\ninput: \n" );
            for (int j = 0; j < BLOCK_W * DPAS_INPUT_CHANNELS; j++)
            {
                    printf("%8.3f ",input_row_0(j));	
            }

            printf("\nweights_0: %d\n", weights_offset_0);
            for (int k = 0; k < WEIGHTS_REG_SIZE; k++)
            {
                    printf("%8.3f ",weights_0(k));	
            }
        }
#endif
#if BLOCK_OC >= 16
        vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_1 = load_filter_nchw_data(surface_weights, weights_offset_1);
#endif  
#if BLOCK_OC >= 32
        vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_2 = load_filter_nchw_data(surface_weights, weights_offset_2);
        vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_3 = load_filter_nchw_data(surface_weights, weights_offset_3);
#endif
#if BLOCK_OC >= 40
		vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_4 = load_filter_nchw_data(surface_weights, weights_offset_4);
#endif
#if BLOCK_OC >= 64
		vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_5 = load_filter_nchw_data(surface_weights, weights_offset_5);
		vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_6 = load_filter_nchw_data(surface_weights, weights_offset_6);
		vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_7 = load_filter_nchw_data(surface_weights, weights_offset_7);
#endif
#if BLOCK_OC == 80
        vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_8 = load_filter_nchw_data(surface_weights, weights_offset_8);
        vector<DT_WEIGHTS, WEIGHTS_REG_SIZE> weights_9 = load_filter_nchw_data(surface_weights, weights_offset_9);
#endif

        accu_row_0_oc_0 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_0, weights_0.format<uint32_t>(), input_row_0.format<uint32_t>());
#if BLOCK_OC >= 16
        accu_row_0_oc_1 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_1, weights_1.format<uint32_t>(), input_row_0.format<uint32_t>());
#endif
#if BLOCK_OC >= 32
        accu_row_0_oc_2 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_2, weights_2.format<uint32_t>(), input_row_0.format<uint32_t>());
        accu_row_0_oc_3 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_3, weights_3.format<uint32_t>(), input_row_0.format<uint32_t>());
#endif
#if BLOCK_OC >= 40
		accu_row_0_oc_4 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_4, weights_4.format<uint32_t>(), input_row_0.format<uint32_t>());
#endif
#if BLOCK_OC >= 64
		accu_row_0_oc_5 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_5, weights_5.format<uint32_t>(), input_row_0.format<uint32_t>());
		accu_row_0_oc_6 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_6, weights_6.format<uint32_t>(), input_row_0.format<uint32_t>());
		accu_row_0_oc_7 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_7, weights_7.format<uint32_t>(), input_row_0.format<uint32_t>());
#endif
#if BLOCK_OC == 80
        accu_row_0_oc_8 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_8, weights_8.format<uint32_t>(), input_row_0.format<uint32_t>());
        accu_row_0_oc_9 = cm_dpas<CM_PRECISION_HF, CM_PRECISION_HF, 8, DPAS_RC>(accu_row_0_oc_9, weights_9.format<uint32_t>(), input_row_0.format<uint32_t>());
#endif
        input_offset += (input_dpas_ic_offset_size * sizeof(DT_IN));
        weights_offset_0 += weights_ic_offset_size;
        weights_offset_1 += weights_ic_offset_size;
        weights_offset_2 += weights_ic_offset_size;
        weights_offset_3 += weights_ic_offset_size;
        weights_offset_4 += weights_ic_offset_size;
        weights_offset_5 += weights_ic_offset_size;
        weights_offset_6 += weights_ic_offset_size;
        weights_offset_7 += weights_ic_offset_size;
		weights_offset_8 += weights_ic_offset_size;
		weights_offset_9 += weights_ic_offset_size;
    }

    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_0);
#if BLOCK_OC >= 16
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_1 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_1);
#endif

#if BLOCK_OC >= 32
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_2 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_2);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_3 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_3);
#endif
#if BLOCK_OC >= 40
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_4 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_4);
#endif
#if BLOCK_OC >= 64
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_5 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_5);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_6 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_6);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_7 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_7);
#endif
#if BLOCK_OC == 80
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_8 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_8);
    vector<DT_OUT, ACCU_REG_SIZE> output_row_0_oc_9 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0_oc_9);
#endif

#if USE_BIAS
	vector<DT_OUT, BLOCK_OC> bias = load_bias(surface_bias, oc_chunk_id * BLOCK_W * sizeof(DT_OUT));
#if DEBUG_MODE

    if (thg_0 == 0 && h_chunk_id == 0 && thread_id_2 == 0)
    {
        printf("bias: \n");
        for (int i = 0; i < BLOCK_OC; i++)
        {
            printf("%8.3f ", bias(i));
        }
    }
#endif // DEBUG_MODE

	#pragma unroll
	for(int bw = 0; bw < BLOCK_W; bw++)
	{
		output_row_0_oc_0.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(0);
#if BLOCK_OC >= 16
		output_row_0_oc_1.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(8);
#endif
#if BLOCK_OC >= 32
		output_row_0_oc_2.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(16);
		output_row_0_oc_3.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(24);
#endif
#if BLOCK_OC >= 40
		output_row_0_oc_4.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(32);
#endif
#if BLOCK_OC >= 64
		output_row_0_oc_5.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(40);
		output_row_0_oc_6.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(48);
		output_row_0_oc_7.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(56);
#endif
#if BLOCK_OC >= 80
		output_row_0_oc_8.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(64);
		output_row_0_oc_9.select<BLOCK_W, 1>(bw * BLOCK_W) += bias.select<BLOCK_W, 1>(72);
#endif
	}
#endif 
  
    const uint output_batch_offset = batch_id * OUTPUT_HEIGHT * OUTPUT_WIDTH * OUTPUT_CHANNELS;
    const uint output_oc_chunk_offset = oc_chunk_id * DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH;
    const uint output_w_chunk_offset = w_chunk_id * BLOCK_W;
    const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * OUTPUT_WIDTH;
    uint32_t output_offset = (output_batch_offset + output_oc_chunk_offset + output_h_chunk_offset + output_w_chunk_offset) * sizeof(DT_OUT);
	
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_0, output_offset, w_chunk_id);  
    
#if BLOCK_OC >= 16
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_1, output_offset, w_chunk_id); 
#endif

#if BLOCK_OC >= 32
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_2, output_offset, w_chunk_id); 
    
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_3, output_offset, w_chunk_id); 
#endif

#if BLOCK_OC >= 40
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_4, output_offset, w_chunk_id); 
#endif

#if BLOCK_OC >= 64
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_5, output_offset, w_chunk_id); 
	
	output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_6, output_offset, w_chunk_id); 
    
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_7, output_offset, w_chunk_id);
#endif

#if BLOCK_OC == 80
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_8, output_offset, w_chunk_id); 
    
    output_offset += (DPAS_OUTPUT_CHANNELS * OUTPUT_HEIGHT * OUTPUT_WIDTH) * sizeof(DT_OUT);
    store_output_wc8_as_nchw<BLOCK_W>(surface_output, output_row_0_oc_9, output_offset, w_chunk_id); 
#endif
}