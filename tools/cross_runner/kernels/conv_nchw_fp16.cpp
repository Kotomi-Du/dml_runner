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

#define UNIT_VAL_ONE 1.0f
#define UNIT_VAL_ZERO 0.0f

#define BLOCK_H 1
#define BLOCK_W 4
#define BLOCK_BATCH 1
#define BLOCK_OC 8

#if !CM_HAS_LSC
#error [Error_device_no_lsc] Kernel designed to use lsc. Current device does not support lsc.
#endif

#if(CM_GENX >= 1280)
#error [Error_device_not_supported] Kernel is not designed for this architecutre.
#endif

#if BLOCK_W > 8
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_w in range: <1; 7>;
#endif

#if BLOCK_OC != 8 && BLOCK_OC != 16 && BLOCK_OC != 32
#error [Error_kernel_config_unsupported_block_w] Kernel designed to with with block_oc which is equal to 8 or 16 or 32;
#endif

#if INPUT_CHANNELS >= 16
#error [Error_input_channel_count] This kernel was designed to work with small input channels, which are not fitting dpas scenarios.
#endif

#define DT_OUT DT_ACCU
#define DT_IN DT_ACCU
#define DT_WEIGHTS DT_ACCU
#define OUTPUT_ELEMENT_SIZE (sizeof(DT_OUT))

#define INPUT_REG_W (BLOCK_W * STRIDE_W + KERNEL_SIZE - 1)
#define INPUT_REG_SIZE (INPUT_REG_W* INPUT_CHANNELS)
#define INPUT_ELEMENT_SIZE (sizeof(DT_IN))
#define WEIGHT_ELEMENT_SIZE (sizeof(DT_WEIGHTS))
#define MAX_ELEMENT_SIZE (sizeof(float))

#define ACCU_REG_SIZE (BLOCK_OC * BLOCK_W)

#define INPUT_ELEMENT_SIZE_I32 int32_t(INPUT_ELEMENT_SIZE)

#define ACTIVATION_LOGISTIC 	41
#define ACTIVATION_RELU		  	32
#define ACTIVATION_LEAKY_RELU 	33
#define MATH_E 2.718281828459045235360287471352f

static const uint32_t output_linear_init_offsets[] = {
                                                0 * OUTPUT_ELEMENT_SIZE,
                                                1 * OUTPUT_ELEMENT_SIZE,
                                                2 * OUTPUT_ELEMENT_SIZE,
                                                3 * OUTPUT_ELEMENT_SIZE,
                                                4 * OUTPUT_ELEMENT_SIZE,
                                                5 * OUTPUT_ELEMENT_SIZE,
                                                6 * OUTPUT_ELEMENT_SIZE,
                                                7 * OUTPUT_ELEMENT_SIZE, 
												8 * OUTPUT_ELEMENT_SIZE, 
											    9 * OUTPUT_ELEMENT_SIZE,
											    10 * OUTPUT_ELEMENT_SIZE,
											    11 * OUTPUT_ELEMENT_SIZE,
											    12 * OUTPUT_ELEMENT_SIZE,
											    13 * OUTPUT_ELEMENT_SIZE,
											    14 * OUTPUT_ELEMENT_SIZE,
											    15 * OUTPUT_ELEMENT_SIZE
                                            };
											
_GENX_ inline DT_ACCU activation_function(uint32_t activation_type, DT_ACCU input, DT_ACCU m, DT_ACCU n)
{
	if(activation_type == ACTIVATION_LOGISTIC)
	{
		DT_ACCU e_pow_x = cm_pow<DT_ACCU>((DT_ACCU)MATH_E, input);
		return e_pow_x/(e_pow_x + UNIT_VAL_ONE);
	}
	else if(activation_type == ACTIVATION_RELU)
	{
		return (input >= UNIT_VAL_ZERO) ? input : UNIT_VAL_ZERO;
	}
	else if(activation_type == ACTIVATION_LEAKY_RELU)
	{
		return (input >= UNIT_VAL_ZERO) ? input : m * input;
	}
	else
	{
		return input;
	}
}
	
_GENX_ inline vector<DT_ACCU, INPUT_REG_SIZE> load_input_nchw(SurfaceIndex surface [[type("buffer_t")]], uint32_t input_width, uint32_t input_height, uint32_t input_pad, uint32_t w_offset, int32_t h_offset, uint32_t batch_base_offset_bytes)
{
	const uint32_t LINEAR_LOAD_SIZE = 16;
	const int32_t h_offset_pad = h_offset - int32_t(input_pad);
	vector<DT_ACCU, INPUT_REG_SIZE> ret(0.0f);
	vector<uint8_t, LINEAR_LOAD_SIZE> predicate(0);
	vector<int32_t, LINEAR_LOAD_SIZE> offsets;
	
	#pragma unroll
	for(int i = 0; i < LINEAR_LOAD_SIZE; i++)
	{
		offsets[i] = (i - int32_t(input_pad)) * INPUT_ELEMENT_SIZE_I32;
	}
	
	offsets += w_offset * INPUT_ELEMENT_SIZE; 	// offset by X
	
	// update predicate mask for left and right padding
	predicate.merge(1, offsets >= 0 & offsets < (input_width * INPUT_ELEMENT_SIZE));
	// update predicate mask for top and bottom padding
	vector<int32_t, LINEAR_LOAD_SIZE> is_not_in_height_range(h_offset_pad < 0 |  h_offset_pad > (input_height - 1));
	predicate.merge(0, is_not_in_height_range);
	
	offsets += h_offset_pad * input_width * INPUT_ELEMENT_SIZE;
	vector<uint32_t, LINEAR_LOAD_SIZE> offsets_u32 = offsets;
	offsets_u32 += batch_base_offset_bytes;
	#pragma unroll
	for(int i = 0; i < INPUT_CHANNELS; i++)
	{
		vector<DT_IN, LINEAR_LOAD_SIZE> load_chunk = cm_load<DT_IN, VectorSize::N1, DataSize::Default, CacheHint::Default, CacheHint::Default>(surface, offsets_u32);
		vector<DT_ACCU, LINEAR_LOAD_SIZE> load_chunk_accu_dt = vector<DT_ACCU, LINEAR_LOAD_SIZE>(load_chunk);
		ret.select<INPUT_REG_W, 1>(i * INPUT_REG_W).merge(load_chunk_accu_dt.select<INPUT_REG_W, 1>(), predicate.select<INPUT_REG_W, 1>());
		offsets_u32 += (input_width * input_height * INPUT_ELEMENT_SIZE);
	}
	
	return ret;
}


_GENX_ inline vector<DT_ACCU, BLOCK_OC * INPUT_CHANNELS> load_weights(SurfaceIndex surface [[type("buffer_t")]], uint32_t kw, uint32_t kh, uint32_t oc_chunk)
{
	/*
		This function requires weights to be in optimal format:  OYXI_o8  or OYXI_o16  (depending on the oc_block)  (oc_block == simd size of the "mad" instructions)
	*/
	vector<DT_ACCU, BLOCK_OC * INPUT_CHANNELS> ret;
	uint32_t offset = (oc_chunk * KERNEL_SIZE * KERNEL_SIZE + kh * KERNEL_SIZE + kw) * INPUT_CHANNELS * BLOCK_OC * WEIGHT_ELEMENT_SIZE;
	const uint32_t BLOCK_SC = MAX_ELEMENT_SIZE/WEIGHT_ELEMENT_SIZE;
	#pragma unroll
	for(int i = 0; i < INPUT_CHANNELS; i++)
	{
		vector<uint32_t, BLOCK_OC/BLOCK_SC> typed_load = cm_load<uint32_t, BLOCK_OC/BLOCK_SC, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface, offset);	
		vector_ref<DT_WEIGHTS, BLOCK_OC> load_data = typed_load.format<DT_WEIGHTS>();
		ret.select<BLOCK_OC, 1>(i * BLOCK_OC) = vector<DT_ACCU, BLOCK_OC>(load_data);
				
		
		offset += BLOCK_OC * WEIGHT_ELEMENT_SIZE;
	}

	return ret;
}


_GENX_ inline void store_output_wc8_as_nchw(SurfaceIndex surface [[type("buffer_t")]], vector_ref<DT_OUT, ACCU_REG_SIZE> grf_chunk, uint32_t output_width, uint32_t output_height, uint32_t byte_offset, uint32_t w_chunk_id)
{   
	const uint32_t output_nchw_plane_size = (output_width * output_height * sizeof(DT_OUT));
	const uint32_t BLOCK_SC = MAX_ELEMENT_SIZE/OUTPUT_ELEMENT_SIZE;
	// first: check if block stores can be used (address aligned to u32)
	// second check if scattared cache-friendly writes can be used 
	// third: use generic, slowest path with scattared, partial writes

	if (sizeof(DT_OUT) == MAX_ELEMENT_SIZE && output_width == 1)
	{	
		vector<DT_OUT, BLOCK_OC> grf_chunk_store = grf_chunk.select<BLOCK_OC, 1>(0); 
		cm_store<uint32_t, BLOCK_OC/BLOCK_SC>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
	}
	else if ((output_width * OUTPUT_ELEMENT_SIZE) % (BLOCK_W * OUTPUT_ELEMENT_SIZE / sizeof(uint32_t)) == 0)
	{
		#pragma unroll
		for(int i = 0; i < BLOCK_OC; i++)
		{
			// pick data to store
			vector<DT_OUT, BLOCK_W> grf_chunk_store = grf_chunk.select<BLOCK_W, BLOCK_OC>(i);                  
			cm_store<uint32_t, BLOCK_W/BLOCK_SC>(surface, byte_offset, grf_chunk_store.format<uint32_t>());
			byte_offset += output_width * output_height * OUTPUT_ELEMENT_SIZE;
		}		
	}
	else if constexpr(BLOCK_W == 8 || BLOCK_W == 16)
	{
		vector<uint32_t, BLOCK_W> offsets(output_linear_init_offsets);
		offsets += byte_offset;
		#pragma unroll
		for(int i = 0; i < BLOCK_OC; i++)
		{
			// pick data to store
			vector<DT_OUT, BLOCK_W> grf_chunk_store = grf_chunk.select<BLOCK_W, BLOCK_OC>(i);                  
			cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store/*, predicate*/);
			offsets += output_nchw_plane_size;
		}
	}
	else
	{
		vector<uint32_t, BLOCK_OC> offsets;
		#pragma unroll
		for(int i = 0; i < BLOCK_OC; i++)
		{
			offsets[i] = i * output_nchw_plane_size + byte_offset;
		}
		
		#pragma unroll
		for(int i = 0; i < BLOCK_W; i++)
		{
			// pick data to store
			vector<DT_OUT, BLOCK_OC> grf_chunk_store = grf_chunk.select<BLOCK_OC, 1>(i * BLOCK_OC);                  
			cm_store<half, VectorSize::N1, DataSize::Default, CacheHint::WriteBack, CacheHint::WriteBack>(surface, offsets, grf_chunk_store/*, predicate*/);
			offsets += OUTPUT_ELEMENT_SIZE;
		}
	}
}

extern "C" _GENX_MAIN_ void convolution_nchw_nondpas(
	SurfaceIndex surface_input [[type("buffer_t")]],
	SurfaceIndex surface_weights [[type("buffer_t")]],
#if USE_BIAS
	SurfaceIndex surface_bias [[type("buffer_t")]],
#endif
	SurfaceIndex surface_constants [[type("buffer_t")]],
	SurfaceIndex surface_output [[type("buffer_t")]]
)
{
    vector<uint32_t, 16> constants = cm_load<uint32_t, 16, DataSize::Default, CacheHint::Cached, CacheHint::Cached>(surface_constants, 0);
	const uint32_t input_height = constants[0];
	const uint32_t input_width = constants[1];
	const uint32_t input_pad = constants[2];
	const uint32_t output_channels = constants[3];
	const uint32_t output_height = constants[4];
	const uint32_t output_width = constants[5];
	const uint32_t stride_h = constants[6];
	const uint32_t activation_type = constants[9];
	const uint32_t activation_alpha = constants[10];
	const uint32_t activation_beta = constants[11];
	
	const uint32_t thg_0 = cm_group_id(0) * cm_local_size(0) + cm_local_id(0);
    const uint32_t thg_1 = cm_group_id(1) * cm_local_size(1) + cm_local_id(1);
    const uint32_t thg_2 = cm_group_id(2) * cm_local_size(2) + cm_local_id(2);   

	const uint32_t w_chunk_id = thg_0;
	const uint32_t h_chunk_id = thg_1;
	const uint32_t thread_per_full_oc = output_channels / BLOCK_OC;
	const uint32_t batch_id = thg_2 / thread_per_full_oc;
	const uint32_t oc_chunk_id = thg_2 % thread_per_full_oc;

    const uint32_t input_batch_offset = batch_id * BLOCK_BATCH * input_width * input_height * INPUT_CHANNELS * sizeof(DT_IN);
    const uint32_t input_w_chunk_offset = w_chunk_id * BLOCK_W * STRIDE_W;
    const uint32_t input_h_chunk_offset = h_chunk_id * BLOCK_H * stride_h;
	
	matrix<DT_ACCU, BLOCK_BATCH, ACCU_REG_SIZE> accu_row_0(0.0f);
	
	#pragma unroll
	for(int kh = 0; kh < KERNEL_SIZE; kh++)
	{
		matrix<DT_ACCU, BLOCK_BATCH, INPUT_REG_SIZE> input_0;
		#pragma unroll
		for(int b = 0; b < BLOCK_BATCH; b++)
		{
			input_0.row(b) = load_input_nchw(surface_input, input_width, input_height, input_pad, input_w_chunk_offset, input_h_chunk_offset + kh, input_batch_offset + b * input_width * input_height * INPUT_CHANNELS * sizeof(DT_IN));
		}

		#pragma unroll
		for(int kw = 0; kw < KERNEL_SIZE; kw++)
		{
			vector<DT_ACCU, BLOCK_OC * INPUT_CHANNELS> weights_chunk_oc_ic = load_weights(surface_weights, kw, kh, oc_chunk_id);
			#pragma unroll
			for(int i = 0; i < INPUT_CHANNELS; i++)
			{
				matrix_ref<DT_ACCU, BLOCK_BATCH, BLOCK_W * STRIDE_W> input_chunk_0 = input_0.select<BLOCK_BATCH, 1, BLOCK_W * STRIDE_W, 1>(0, kw + i * INPUT_REG_W);
				vector_ref<DT_ACCU, BLOCK_OC> weights_chunk_ic = weights_chunk_oc_ic.select<BLOCK_OC, 1>(i * BLOCK_OC);
		
				#pragma unroll
				for(int b = 0; b < BLOCK_BATCH; b++)
				{
					#pragma unroll
					for(int bw = 0; bw < BLOCK_W; bw++)
					{
						// as long as accumulator, input and weights are the same data type this will compile into single mad instruction				
						accu_row_0.select<1, 1, BLOCK_OC, 1>(b, bw * BLOCK_OC) += input_chunk_0.select<1, 1, 1, 1>(b, bw * STRIDE_W).replicate<BLOCK_OC>() * weights_chunk_ic;
					}
				}
			}		
		}		
	}
	
	#pragma unroll
	for(int b = 0; b < BLOCK_BATCH; b++)
	{
		#pragma unroll
		for(int bw = 0; bw < ACCU_REG_SIZE; bw++)
		{
            DT_ACCU activation_input = accu_row_0[b][bw];
			accu_row_0.select<1, 1, 1, 1>(b, bw) = activation_function(activation_type, activation_input, (float)activation_alpha, (float)activation_beta);
        }
    }
	// if the DT_OUT == DT_ACCU then compiler will not do anything here
	// but if data types are different then this cast accumulator to output type
    //vector<DT_OUT, ACCU_REG_SIZE> output_row_0 = vector<DT_OUT, ACCU_REG_SIZE>(accu_row_0);
    matrix<DT_OUT, BLOCK_BATCH, ACCU_REG_SIZE> output_row_0 = matrix<DT_OUT, BLOCK_BATCH, ACCU_REG_SIZE>(accu_row_0);
	
	const uint output_batch_offset = batch_id * BLOCK_BATCH * output_height * output_width * output_channels;
    const uint output_oc_chunk_offset = oc_chunk_id * BLOCK_OC * output_height * output_width;
    const uint output_w_chunk_offset = w_chunk_id * BLOCK_W;
    const uint output_h_chunk_offset = h_chunk_id * BLOCK_H * output_width;
    uint32_t output_offset = (output_batch_offset + output_oc_chunk_offset + output_h_chunk_offset + output_w_chunk_offset) * sizeof(DT_OUT);
	
	store_output_wc8_as_nchw(surface_output, output_row_0.row(0), output_width, output_height, output_offset, w_chunk_id);
	//store_output_wc8_as_nchw(surface_output, output_row_0.row(1), output_width, output_height, output_offset + output_height * output_width * output_channels * sizeof(DT_OUT), w_chunk_id);
}
