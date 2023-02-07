#include "dx12_utils.h"
#include "gemm.h"
#include "conv.h"
#include "softmax.h"
#include "layers_utils.h"

#include <iostream>
#include <optional>
#include <span>
#include <format>
#include <random>
#include <chrono>

#include "CLI/App.hpp"
#include "CLI/Formatter.hpp"
#include "CLI/Config.hpp"


template<typename TimeType>
inline void print_performance_stats(const std::vector<TimeType>& timings)
{
    TimeType avg(0);
    TimeType best((std::numeric_limits<uint32_t>::max)());
    TimeType median(0);

    // avg and best
    {
        for (const auto& t : timings)
        {
            avg += t;
            if (t < best)
            {
                best = t;
            }
        }
        avg /= timings.size();
    }

    // median
    {
        auto timings_copy = timings;
        std::nth_element(timings_copy.begin(), timings_copy.begin() + timings_copy.size() / 2, timings_copy.end());
        median = timings_copy[timings_copy.size() / 2];
    }

    std::cout << "Avg: " << avg << std::endl;
    std::cout << "Median: " << avg << std::endl;
    std::cout << "Best: " << best << std::endl;
}

inline void randomize_linear_container_float(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = float;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = static_cast<Dt>(dist(gen));
    }
}


inline void randomize_linear_container_half(std::mt19937& gen, std::uniform_real_distribution<float>& dist, std::span<std::byte> container)
{
    using Dt = Half;
    auto* ptr = reinterpret_cast<Dt*>(container.data());
    for (auto i = 0; i < container.size() / sizeof(Dt); i++)
    {
        ptr[i] = DirectX::PackedVector::XMConvertFloatToHalf(dist(gen));
    }
}

inline void add_data_type_cli_option(CLI::App* opts, std::string_view opt_name, DataType& dt)
{
    opts->add_option("--data_type", dt)->required()->check(CLI::IsMember({ DataType::eFp32, DataType::eFp16 }))
        ->transform(CLI::Transformer(std::map<std::string, DataType>{
            {"fp32", DataType::eFp32}, { "fp16", DataType::eFp16 }
    }, CLI::ignore_case, CLI::ignore_underscore));
}

inline void add_data_layout_cli_option(CLI::App* opts, std::string_view opt_name, DataLayout& layout)
{
    opts->add_option("--layout", layout)->required()->check(CLI::IsMember({ DataLayout::eNCHW, DataLayout::eNHWC }))
        ->transform(CLI::Transformer(std::map<std::string, DataLayout>{
            {"nchw", DataLayout::eNCHW}, { "nhwc", DataLayout::eNHWC }, 
    }, CLI::ignore_case, CLI::ignore_underscore));
}

struct ConformanceResult
{
    bool passed = true;
    float epsilon = 0.0f;
    float biggest_difference = 0.0f;
    float node_value = 0.0f;
    float reference_value = 0.0f;
    std::uint32_t index = 0;
    std::size_t tested_samples_count = 0;
};

inline float cast_to_float(Half v)
{
    return DirectX::PackedVector::XMConvertHalfToFloat(v);
}

inline float cast_to_float(float v)
{
    return v;
}

template<typename Dt>
inline ConformanceResult run_conformance_check(const std::vector<std::byte>& gpu_untyped_result, const std::vector<std::byte>& dnnl_untyped_result, float epsilon)
{
    const auto* gpu_typed_result = reinterpret_cast<const Dt*>(gpu_untyped_result.data());
    const auto* dnnl_typed_result = reinterpret_cast<const Dt*>(dnnl_untyped_result.data());

    // compare results
    ConformanceResult ret;
    ret.epsilon = epsilon;
    for (std::uint32_t i = 0; i < gpu_untyped_result.size() / sizeof(Dt); i++)
    {
        ret.node_value = cast_to_float(gpu_typed_result[i]);
        ret.reference_value = cast_to_float(dnnl_typed_result[i]);

        const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

        if (abs_diff > ret.epsilon)
        {
            ret.passed = false;

            std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: {} \n", ret.node_value, ret.reference_value, i, abs_diff);
        }
        ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
        ret.tested_samples_count++;
    }
    return ret;
}

enum class NodeType
{
    eGemm,
    eConv,
    eSoftmax,
    eCount
};

class NodeDispatcher
{
public:
    virtual std::uint32_t get_total_descriptor_count() = 0;
    virtual void initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) = 0;
    virtual void execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list) = 0;

    virtual ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) = 0;

    virtual ~NodeDispatcher() = default;
};

class GemmDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        std::uint32_t M;
        std::uint32_t K;
        std::uint32_t N;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            opts->add_option("M", params.M)->required();
            opts->add_option("K", params.K)->required();
            opts->add_option("N", params.N)->required();
        }
    };
public:
    GemmDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , gemm_(dml::TensorDimensions{ 1, 1, params_.M, params_.K }, dml::TensorDimensions{ 1, 1, params_.K, params_.N }, dml_device, d3d12_device)
        , d3d12_device_(d3d12_device)
        , input_data_a_(params_.M * params_.K)
        , input_data_b_(params_.K * params_.N)
    {
        const auto tensor_a_desc = gemm_.get_tensor_a_desc();
        const auto tensor_b_desc = gemm_.get_tensor_b_desc();
        const auto tensor_out_desc = gemm_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_a_desc.totalTensorSizeInBytes;
        const auto tensor_b_bytes_width = tensor_b_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;


        upload_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width + tensor_b_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_a_ = create_buffer(d3d12_device, tensor_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        input_buffer_b_ = create_buffer(d3d12_device, tensor_b_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);

        for (auto& a : input_data_a_)
        {
            a = uniform_distribution(random_generator);
        }
        for (auto& b : input_data_b_)
        {
            b = uniform_distribution(random_generator);
        }

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::memcpy(upload_mapped_ptr, input_data_a_.data(), tensor_a_bytes_width);
        std::memcpy(upload_mapped_ptr + tensor_a_bytes_width, input_data_b_.data(), tensor_b_bytes_width);
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);


        cmd_list->CopyBufferRegion(input_buffer_a_.Get(), 0, upload_buffer_.Get(), 0, tensor_a_bytes_width);
        cmd_list->CopyBufferRegion(input_buffer_b_.Get(), 0, upload_buffer_.Get(), tensor_a_bytes_width, tensor_b_bytes_width);

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_a_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_b_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return gemm_.get_total_descriptor_count();
    }

    void initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        gemm_.create_binding_tables(cpu_handle, gpu_handle);

        // Record execution of the operator initializer.
        gemm_.record_initialize(dml_cmd_recorder, cmd_list);
    }

    void execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
    {
        gemm_.record_execute(dml_cmd_recorder, cmd_list, output_buffer_.Get(), input_buffer_a_.Get(), input_buffer_b_.Get());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
    {
        const auto tensor_a_desc = gemm_.get_tensor_a_desc();
        const auto tensor_b_desc = gemm_.get_tensor_b_desc();
        const auto tensor_out_desc = gemm_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_a_desc.totalTensorSizeInBytes;
        const auto tensor_b_bytes_width = tensor_b_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;

        // readback data and validate
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> data_out(tensor_out_bytes_width);
        std::byte* readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
        readback_buffer->Unmap(0, nullptr);

        std::vector<float> cpu_data_f32(params_.M * params_.N);
        cpu_op::gemm<float>(params_.M, params_.K, params_.N, 1.0f, 1.0f, input_data_a_.data(), input_data_b_.data(), nullptr, cpu_data_f32.data());

        const auto* gpu_data_out_f32 = reinterpret_cast<const float*>(data_out.data());
        // compare results
        ConformanceResult ret;
        ret.epsilon = 0.001f;
        for (std::uint32_t i = 0; i < params_.M * params_.N; i++)
        {
            ret.node_value = gpu_data_out_f32[i];
            ret.reference_value = cpu_data_f32[i];
            const auto abs_diff = std::abs(ret.node_value - ret.reference_value);

            if (abs_diff > ret.epsilon)
            {
                ret.passed = false;

                std::cout << std::format("Mismatch, gpu: {}, cpu: {}, at index: {}. Absolute differece: \n", ret.node_value, ret.reference_value, i, abs_diff);
            }
            ret.biggest_difference = std::max(ret.biggest_difference, abs_diff);
            ret.tested_samples_count++;
        }
        return ret;
    }

private:
    create_params_t params_;
    gpu_op::Gemm gemm_;
    ID3D12Device* d3d12_device_;

    std::vector<float> input_data_a_;
    ComPtr<ID3D12Resource> input_buffer_a_;
    std::vector<float> input_data_b_;
    ComPtr<ID3D12Resource> input_buffer_b_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class ConvolutionBaseDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        std::uint32_t batch;
        std::uint32_t ic;
        std::uint32_t oc;
        std::uint32_t in_width;
        std::uint32_t in_height;
        std::uint32_t in_pad;
        std::uint32_t out_pad;
        std::uint32_t kernel_size;
        std::uint32_t stride;
        bool no_bias = false;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt);
            add_data_layout_cli_option(opts, "--layout", params.layout);
            opts->add_option("--batch", params.batch)->required();
            opts->add_option("--ic", params.ic)->required();
            opts->add_option("--oc", params.oc)->required();
            opts->add_option("--in_width", params.in_width)->required();
            opts->add_option("--in_height", params.in_height)->required();
            opts->add_option("--in_pad", params.in_pad)->required();
            opts->add_option("--out_pad", params.out_pad)->required();
            opts->add_option("--kernel_size", params.kernel_size)->required();
            opts->add_option("--stride", params.stride)->required();
            opts->add_flag("--no_bias", params.no_bias);
        }
    };

    ConvolutionBaseDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , d3d12_device_(d3d12_device)
        , input_data_(params_.batch* params_.ic* params_.in_height* params_.in_width * get_data_type_bytes_width(params_.dt))
        , filter_data_(params_.oc* params_.ic* params_.kernel_size* params_.kernel_size * get_data_type_bytes_width(params_.dt))

    {
        if (!params_.no_bias)
        {
            bias_data_ = std::vector<std::byte>(params_.oc * get_data_type_bytes_width(params_.dt));
        }
        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(-0.5f, 0.5f);
        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_);
            randomize_linear_container_float(random_generator, uniform_distribution, filter_data_);
            if (use_bias())
            {
                randomize_linear_container_float(random_generator, uniform_distribution, bias_data_);
            }
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_);
            randomize_linear_container_half(random_generator, uniform_distribution, filter_data_);
            if (use_bias())
            {
                randomize_linear_container_half(random_generator, uniform_distribution, bias_data_);
            }
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        const auto tensor_input_bytes_width = input_data_.size();
        const auto tensor_filter_bytes_width = filter_data_.size();
        const auto tensor_bias_bytes_width = bias_data_.size();
        const auto out_width = (params_.in_width - params_.kernel_size + params_.in_pad + params_.in_pad) / params_.stride + 1;
        const auto out_height = (params_.in_height - params_.kernel_size + params_.in_pad + params_.in_pad) / params_.stride + 1;
        const auto tensor_out_bytes_width = params_.batch * params_.oc * out_height * out_width * get_data_type_bytes_width(params_.dt);

        upload_buffer_ = create_buffer(d3d12_device_, tensor_input_bytes_width + tensor_filter_bytes_width + tensor_bias_bytes_width,
            D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_input_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        filter_buffer_ = create_buffer(d3d12_device, tensor_filter_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);      
        if (use_bias())
        {
            bias_buffer_ = create_buffer(d3d12_device, tensor_bias_bytes_width,
                D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        }
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::size_t memcopy_offset = 0;
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        std::memcpy(upload_mapped_ptr + memcopy_offset, filter_data_.data(), tensor_filter_bytes_width);
        if (use_bias())
        {
            memcopy_offset += tensor_filter_bytes_width;
            std::memcpy(upload_mapped_ptr + memcopy_offset, bias_data_.data(), tensor_bias_bytes_width);
        }
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        memcopy_offset = 0;
        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_input_bytes_width);
        memcopy_offset += tensor_input_bytes_width;
        cmd_list->CopyBufferRegion(filter_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_filter_bytes_width);
        if (use_bias())
        {
            memcopy_offset += tensor_filter_bytes_width;
            cmd_list->CopyBufferRegion(bias_buffer_.Get(), 0, upload_buffer_.Get(), memcopy_offset, tensor_bias_bytes_width);
        }

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(filter_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        if (use_bias())
        {
            barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(bias_buffer_.Get(),
                D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        }
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list) override
    {
        const auto tensor_out_bytes_width = output_buffer_->GetDesc().Width;

        // readback data and validate
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> data_out(tensor_out_bytes_width);
        std::byte* readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
        readback_buffer->Unmap(0, nullptr);

        const auto dnnl_untyped_result = get_dnnl_result();

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, dnnl_untyped_result, 0.001f);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, dnnl_untyped_result, 0.05f);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;
    }

protected:
    inline bool use_bias() const
    {
        return !params_.no_bias;
    }

    std::vector<std::byte> get_dnnl_result() const
    {
        cpu_op::bindings_t bindings{};
        {
            bindings.input.data = input_data_.data();
            bindings.input.dt = params_.dt;
            bindings.input.layout = params_.layout;
            bindings.input.dims = { params_.batch, params_.ic, params_.in_width, params_.in_height };
        }

        {
            bindings.filter.data = filter_data_.data();
            bindings.filter.dt = params_.dt;
            bindings.filter.layout = params_.layout;
            bindings.filter.dims = { params_.oc , params_.ic, params_.kernel_size, params_.kernel_size };
        }
        if(use_bias())
        {
            bindings.bias.data = bias_data_.data();
            bindings.bias.dt = params_.dt;
            bindings.bias.layout = params_.layout;
            bindings.bias.dims = { params_.oc , 1, 1, 1 };
        }
        cpu_op::opts_t opts{};
        opts.inp_pad = params_.in_pad;
        opts.out_pad = params_.out_pad;
        opts.stride = params_.stride;
        opts.out_layout = params_.layout;
        opts.out_dt = params_.dt;
        return cpu_op::convolution(bindings, opts);
    }

protected:
    ID3D12Device* d3d12_device_;
    create_params_t params_;

    std::vector<std::byte> input_data_;
    std::vector<std::byte> filter_data_;
    std::vector<std::byte> bias_data_;

    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> filter_buffer_;
    ComPtr<ID3D12Resource> bias_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

class ConvolutionDirectMLDispatcher : public ConvolutionBaseDispatcher
{
public:

    ConvolutionDirectMLDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, ID3D12GraphicsCommandList* cmd_list)
        : ConvolutionBaseDispatcher(std::move(params), d3d12_device, cmd_list)
        , conv_(dml::TensorDimensions{params_.batch, params_.ic, params_.in_width, params_.in_height},
            dml::TensorDimensions{ params_.oc, params_.ic, params_.kernel_size, params_.kernel_size},
            to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout),
            params_.stride, params_.in_pad, params_.out_pad, !params_.no_bias, dml_device, d3d12_device)

    {

    } 

    std::uint32_t get_total_descriptor_count()override
    {
        return conv_.get_total_descriptor_count();
    }

    void initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle) override
    {
        conv_.create_binding_tables(cpu_handle, gpu_handle);
        conv_.record_initialize(dml_cmd_recorder, cmd_list);
    }

    void execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list) override
    {
        conv_.record_execute(dml_cmd_recorder, cmd_list, output_buffer_.Get(), input_buffer_.Get(), filter_buffer_.Get(), bias_buffer_.Get());
    }

private:
    gpu_op::Convolution conv_;

};

class SoftmaxDispatcher : public NodeDispatcher
{
public:
    struct create_params_t
    {
        DataType dt;
        DataLayout layout;
        std::uint32_t batch;
        std::uint32_t ic;
        std::uint32_t in_width;
        std::uint32_t in_height;
        std::uint32_t axis;

        inline static void add_cli_options(CLI::App* opts, create_params_t& params)
        {
            add_data_type_cli_option(opts, "--data_type", params.dt);
            add_data_layout_cli_option(opts, "--layout", params.layout);
            opts->add_option("--batch", params.batch)->required();
            opts->add_option("--ic", params.ic)->required();
            opts->add_option("--in_width", params.in_width)->required();
            opts->add_option("--in_height", params.in_height)->required();
            opts->add_option("--axis", params.axis, "axis represents the axis of which the SoftMax is calculated.")->required();
        }
    };
public:
    SoftmaxDispatcher(create_params_t&& params, ID3D12Device* d3d12_device, IDMLDevice* dml_device, ID3D12GraphicsCommandList* cmd_list)
        : params_(std::move(params))
        , softmax_(params_.axis, dml::TensorDimensions{ params_.batch, params_.ic, params_.in_width, params.in_height },
            to_dml_data_type(params_.dt), to_dml_tensor_policy(params_.layout), dml_device, d3d12_device)
        , d3d12_device_(d3d12_device)
        , input_data_(params_.batch * params_.ic * params_.in_width * params.in_height * get_data_type_bytes_width(params_.dt))
    {
        const auto tensor_in_desc = softmax_.get_tensor_input_desc();
        const auto tensor_out_desc = softmax_.get_tensor_out_desc();
        const auto tensor_a_bytes_width = tensor_in_desc.totalTensorSizeInBytes;
        const auto tensor_out_bytes_width = tensor_out_desc.totalTensorSizeInBytes;


        upload_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_STATE_GENERIC_READ);
        input_buffer_ = create_buffer(d3d12_device, tensor_a_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        output_buffer_ = create_buffer(d3d12_device, tensor_out_bytes_width,
            D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);

        // randomize data
        std::mt19937 random_generator(42); // static, create it once!
        std::uniform_real_distribution<float> uniform_distribution(0.0f, 5.0f);

        if (params_.dt == DataType::eFp32)
        {
            randomize_linear_container_float(random_generator, uniform_distribution, input_data_);
        }
        else if (params_.dt == DataType::eFp16)
        {
            randomize_linear_container_half(random_generator, uniform_distribution, input_data_);
        }
        else
        {
            assert(false && "Unsupported data type in convolution dispatcher!");
        }

        // copy data into buffer
        std::byte* upload_mapped_ptr = nullptr;
        upload_buffer_->Map(0, nullptr, reinterpret_cast<void**>(&upload_mapped_ptr));
        std::memcpy(upload_mapped_ptr, input_data_.data(), tensor_a_bytes_width);
        // unmap memory
        upload_buffer_->Unmap(0, nullptr);

        cmd_list->CopyBufferRegion(input_buffer_.Get(), 0, upload_buffer_.Get(), 0, tensor_a_bytes_width);

        std::vector<CD3DX12_RESOURCE_BARRIER> barriers;
        barriers.push_back(CD3DX12_RESOURCE_BARRIER::Transition(input_buffer_.Get(),
            D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_UNORDERED_ACCESS));
        cmd_list->ResourceBarrier(static_cast<std::uint32_t>(barriers.size()), barriers.data());
    }

    std::uint32_t get_total_descriptor_count() override
    {
        return softmax_.get_total_descriptor_count();
    }

    void initialize(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list, D3D12_CPU_DESCRIPTOR_HANDLE cpu_handle, D3D12_GPU_DESCRIPTOR_HANDLE gpu_handle)
    {
        softmax_.create_binding_tables(cpu_handle, gpu_handle);
        softmax_.record_initialize(dml_cmd_recorder, cmd_list);
    }

    void execute(IDMLCommandRecorder* dml_cmd_recorder, ID3D12GraphicsCommandList* cmd_list)
    {
        softmax_.record_execute(dml_cmd_recorder, cmd_list, output_buffer_.Get(), input_buffer_.Get());
    }

    ConformanceResult validate_conformance(ID3D12CommandQueue* command_queue,
        ID3D12CommandAllocator* command_allocator, ID3D12GraphicsCommandList* command_list)
    {
        const auto tensor_out_bytes_width = output_buffer_->GetDesc().Width;

        // readback data and validate
        auto readback_buffer = create_buffer(d3d12_device_, tensor_out_bytes_width, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_STATE_COPY_DEST);
        auto readback_output_barrirer = CD3DX12_RESOURCE_BARRIER::Transition(output_buffer_.Get(),
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list->ResourceBarrier(1, &readback_output_barrirer);
        command_list->CopyResource(readback_buffer.Get(), output_buffer_.Get());
        close_execute_reset_wait(d3d12_device_, command_queue, command_allocator, command_list);

        std::vector<std::byte> data_out(tensor_out_bytes_width);
        std::byte* readback_mapped_ptr = nullptr;
        readback_buffer->Map(0, nullptr, reinterpret_cast<void**>(&readback_mapped_ptr));
        std::memcpy(data_out.data(), readback_mapped_ptr, data_out.size());
        readback_buffer->Unmap(0, nullptr);

        const auto dnnl_untyped_result = cpu_op::softmax(params_.axis, input_data_.data(),
            { params_.batch, params_.ic, params_.in_height, params_.in_width }, params_.dt, params_.layout);

        if (params_.dt == DataType::eFp32)
        {
            return run_conformance_check<float>(data_out, dnnl_untyped_result, 0.001f);
        }
        else if (params_.dt == DataType::eFp16)
        {
            return run_conformance_check<Half>(data_out, dnnl_untyped_result, 0.05f);
        }
        assert(false && "Unsupported output data type!");
        ConformanceResult ret{};
        return ret;
    }


private:
    create_params_t params_;
    gpu_op::Softmax softmax_;
    ID3D12Device* d3d12_device_;

    std::vector<std::byte> input_data_;

    ComPtr<ID3D12Resource> input_buffer_;
    ComPtr<ID3D12Resource> output_buffer_;
    ComPtr<ID3D12Resource> upload_buffer_;
};

struct CliOptions
{
    NodeType node_type = NodeType::eCount;
    std::uint32_t dispatch_iterations = 1;
    bool no_conformance_check = false;

    GemmDispatcher::create_params_t gemm_opts{};
    ConvolutionBaseDispatcher::create_params_t conv_opts{};
    SoftmaxDispatcher::create_params_t softmax_opts{};
};

int main()
{
    constexpr const std::uint32_t MAX_ITERATIONS = 10'000;

    CliOptions opts;
    CLI::App dml_runner_app{ "App to microbenchmark and developer dml kernels.", "DirectML runner." };
    dml_runner_app.add_option("--type", opts.node_type, "Name of the type of layer to run.")
        ->required()->check(CLI::IsMember({NodeType::eConv, NodeType::eGemm, NodeType::eSoftmax }))->
        transform(CLI::Transformer(std::map<std::string, NodeType>{
            { "conv", NodeType::eConv },
            { "gemm", NodeType::eGemm },
            { "softmax", NodeType::eSoftmax }
    }, CLI::ignore_case, CLI::ignore_underscore));
    dml_runner_app.add_option("--iters", opts.dispatch_iterations, "How many iterations to run.")->check(CLI::Range(1u, MAX_ITERATIONS));
    dml_runner_app.add_flag("--no_conform", opts.no_conformance_check);

    auto gemm_option_groups = dml_runner_app.add_subcommand("gemm_opts", "Options for genn layer.");
    GemmDispatcher::create_params_t::add_cli_options(gemm_option_groups, opts.gemm_opts);
    auto conv_option_groups = dml_runner_app.add_subcommand("conv_opts", "Options for convolution layer.");
    ConvolutionBaseDispatcher::create_params_t::add_cli_options(conv_option_groups, opts.conv_opts);
    auto softmax_option_groups = dml_runner_app.add_subcommand("softmax_opts", "Options for softmax layer.");
    SoftmaxDispatcher::create_params_t::add_cli_options(softmax_option_groups, opts.softmax_opts);

    try {
        dml_runner_app.parse();
    }
    catch (const CLI::ParseError& e) {
        return dml_runner_app.exit(e);
    }

    const auto dumped_config = dml_runner_app.config_to_str(true);
    std::cout << std::format("Running app with config:\n {}", dumped_config);

    assert(opts.node_type != NodeType::eCount);
    if (opts.node_type == NodeType::eConv && !conv_option_groups->parsed())
    {
        std::cout << "Convoltion options not set.\n";
        return -1;
    }
    if (opts.node_type == NodeType::eGemm && !gemm_option_groups->parsed())
    {
        std::cout << "Gemm options not set.\n";
        return -1;
    }
    if (opts.node_type == NodeType::eSoftmax && !softmax_option_groups->parsed())
    {
        std::cout << "Softmax options not set.\n";
        return -1;
    }

    try
    {
        ComPtr<ID3D12Device> d3d12_device;
        ComPtr<ID3D12CommandQueue> command_queue;
        ComPtr<ID3D12CommandAllocator> command_allocator;
        ComPtr<ID3D12GraphicsCommandList> command_list;
        initalize_d3d12(d3d12_device, command_queue, command_allocator, command_list);
        auto dml_device = create_dml_device(d3d12_device.Get());
        assert(opts.dispatch_iterations < MAX_ITERATIONS);
        auto performance_collector = initialize_d3d12_performance_collector(d3d12_device.Get(), MAX_ITERATIONS);

        // The command recorder is a stateless object that records Dispatches into an existing Direct3D 12 command list.
        ComPtr<IDMLCommandRecorder> dml_command_recorder;
        throw_if_failed(dml_device->CreateCommandRecorder(IID_PPV_ARGS(dml_command_recorder.ReleaseAndGetAddressOf())), "create dml command recorder");

        std::unique_ptr<NodeDispatcher> node;
        if (opts.node_type == NodeType::eGemm)
        {
            node = std::make_unique<GemmDispatcher>(std::move(opts.gemm_opts), d3d12_device.Get(), dml_device.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eConv)
        {
            node = std::make_unique<ConvolutionDirectMLDispatcher>(std::move(opts.conv_opts), d3d12_device.Get(), dml_device.Get(), command_list.Get());
        }
        else if (opts.node_type == NodeType::eSoftmax)
        {
            node = std::make_unique<SoftmaxDispatcher>(std::move(opts.softmax_opts), d3d12_device.Get(), dml_device.Get(), command_list.Get());
        }
        else
        {
            assert(false && "Unknown node type!");
        }

        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());
        const auto descriptors_count = node->get_total_descriptor_count();
        
        // bind descriptor heap
        auto descriptor_heap = create_descriptor_heap(d3d12_device.Get(), descriptors_count);
        ID3D12DescriptorHeap* d3d12_descriptor_heaps[] = { descriptor_heap.Get() };
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);


        // initalize
        node->initialize(dml_command_recorder.Get(), command_list.Get(),
            descriptor_heap->GetCPUDescriptorHandleForHeapStart(), descriptor_heap->GetGPUDescriptorHandleForHeapStart());
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        // 
        // Bind and execute the operator on the GPU.
        // 
        // 
        command_list->SetDescriptorHeaps(1, d3d12_descriptor_heaps);

        for (std::uint32_t i = 0; i < opts.dispatch_iterations; ++i)
        {
            performance_collector.add_timestamp(command_list.Get());
            node->execute(dml_command_recorder.Get(), command_list.Get());
            performance_collector.add_timestamp(command_list.Get());
        }
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        const auto device_remove_reason = d3d12_device->GetDeviceRemovedReason();
        if (device_remove_reason != S_OK)
        {
            std::cout << std::format("Device removal. Reason: {}\n", device_remove_reason);
        }

        if (opts.no_conformance_check)
        {
            std::cout << std::format("Skipping conformance check as requested by cmd line.\n");
        }
        else
        {
            const auto conformance_result = node->validate_conformance(command_queue.Get(), command_allocator.Get(), command_list.Get());
            std::cout << std::format("Conformance {}. Tested values (tensor out elements count): {} \n", conformance_result.passed, conformance_result.tested_samples_count);
            std::cout << std::format("Biggest difference in the output tensor: {}. It is in the epsilion range: {}. \n", conformance_result.biggest_difference, conformance_result.epsilon);

            if (!conformance_result.passed)
            {
                return -2;
            }
        }


        // Copy the timing data back
        command_list->ResolveQueryData(
            performance_collector.timestamp_query_heap.Get(),
            D3D12_QUERY_TYPE_TIMESTAMP,
            0,
            performance_collector.timestamp_index,
            performance_collector.timestamp_readback_buffer.Get(),
            0);
        close_execute_reset_wait(d3d12_device.Get(), command_queue.Get(), command_allocator.Get(), command_list.Get());

        uint64_t timestamp_frequency = 0;
        command_queue->GetTimestampFrequency(&timestamp_frequency);

        const auto timestamps_timings = get_timestamps_timings_from_ptr<std::chrono::microseconds>(timestamp_frequency, performance_collector.timestamp_readback, performance_collector.timestamp_index);
        performance_collector.timestamp_index = 0;

        std::vector<std::chrono::microseconds> timings(timestamps_timings.size() / 2);
        for (uint32_t i = 0; i < timings.size(); i++)
        {
            const auto t0 = timestamps_timings[i * 2];
            const auto t1 = timestamps_timings[i * 2 + 1];
            timings[i] = t1 - t0;
        }

        print_performance_stats(timings);
    }
    catch (std::exception e)
    {
        std::cout << std::format("Exception caught: {} \n", e.what());
    }

    return 0;
}