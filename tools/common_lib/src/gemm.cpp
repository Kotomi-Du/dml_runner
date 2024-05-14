#include "gemm.h"
#include "dnnl_utils.h"

std::vector<std::byte> dnnl_gemm_op::gemm(const bindings_t& bindings, opts_t opts)
{
    using namespace dnnl_utils;
    static dnnl::engine engine(dnnl::engine::kind::gpu, 0);
    
    const auto enable_profiling = opts.execution_iterations > 1;
    dnnl::stream stream = [&]()
    {
        auto stream_flags = dnnl::stream::flags::default_flags;
        if (enable_profiling)
        {
            stream_flags |= dnnl::stream::flags::profiling;
        }
        return dnnl::stream(engine, stream_flags);
    }();

    const auto engine_kind = engine.get_kind();
    stream.wait();  // just to be sure we can freely upload the input data   


    dnnl::memory input_a_memory = [&](const auto& binding)
    {
        dnnl::memory ret;
        if (opts.a_transposed)
        {
            dnnl::memory::desc transposed_desc = convert_to_ncwh_format(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt));
            ret = dnnl::memory(transposed_desc, engine);
        }
        else
        {
            ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        }
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_a);

    dnnl::memory input_b_memory = [&](const auto& binding)
    {
        dnnl::memory ret;
        if (opts.b_transposed)
        {
            dnnl::memory::desc transposed_desc = convert_to_ncwh_format(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt));
            ret = dnnl::memory(transposed_desc, engine);
        }
        else
        {
            ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        }
        copy_to_dnnl_memory(ret, binding.data);
        return ret;
    }(bindings.input_b);
   
    dnnl::memory input_c_memory = [&](const auto& binding)
    {
        // if (!binding.data)
        // {
        //     return dnnl::memory{};
        // }
        auto ret = dnnl::memory(to_dnnl_mem_desc(binding.shape, binding.layout, binding.dt), engine);
        if(binding.data)
        {
            copy_to_dnnl_memory(ret, binding.data);
        }
        
        return ret;
    }(bindings.input_c);

    dnnl::post_ops po{};
    dnnl::primitive_attr attrs{};
    if (opts.force_fp32_accumulator)
    {
        attrs.set_accumulation_mode(dnnl::accumulation_mode::strict);
    }

    if (opts.alpha != 1.0f) {
        po.append_eltwise(dnnl::algorithm::eltwise_linear, opts.alpha, 0.0f);

    }

    if (opts.beta != 0.0f) {
        po.append_sum(opts.beta);
    }

    if (opts.activation.type != ActivationType::eUnknown)
    {
        po.append_eltwise(to_dnnl_activation_type(opts.activation.type), opts.activation.alpha, opts.activation.beta);
    }

    attrs.set_post_ops(po);

    dnnl::matmul::primitive_desc matmul_desc(engine,
        input_a_memory.get_desc(),
        input_b_memory.get_desc(),
        //{}, // we dont use bias for c_tensir
        input_c_memory.get_desc(),
        attrs
    );
    const auto guery_impl_str = matmul_desc.impl_info_str();
    std::cout << "ref query impl: " << guery_impl_str << std::endl;

    auto matmul = dnnl::matmul(matmul_desc);
    std::unordered_map<int, dnnl::memory> args;
    args.insert({ DNNL_ARG_SRC, input_a_memory });
    args.insert({ DNNL_ARG_WEIGHTS, input_b_memory });
    args.insert({ DNNL_ARG_DST, input_c_memory });

    std::size_t post_ops_idx = 0ull;

    for (int i = 0; i < opts.execution_iterations; i++)
    {
        matmul.execute(stream, args);
    }

    stream.wait();

    if (enable_profiling)
    {
        const auto profiling_usecs_data = dnnl::get_profiling_data(stream, dnnl::profiling_data_kind::time);
        const auto avg_perf = std::accumulate(profiling_usecs_data.begin(), profiling_usecs_data.end(), 0.0) / profiling_usecs_data.size();
        std::cout << "OneDNN avg performance time: " << (float)avg_perf / 1000.0f << " ms." << std::endl;
    }

    auto* out_dnnl_data = input_c_memory.map_data<uint8_t>();
    assert(out_dnnl_data != nullptr && "[dnnl][gemm] Couldnt map output memory!");

    const auto om_desc = input_c_memory.get_desc();
    const auto copy_size = om_desc.get_size();
    std::vector<std::byte> ret(copy_size);
    std::memcpy(ret.data(), out_dnnl_data, copy_size);
    input_c_memory.unmap_data(out_dnnl_data);
    return ret;
}