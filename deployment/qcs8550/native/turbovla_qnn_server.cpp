// Persistent QNN runner for the fixed TurboVLA LIBERO Object graph split.
// The wire protocol is intentionally the same framed named-array format used
// by the earlier XR runners: host preprocessing stays explicit and testable.

#include <arpa/inet.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "QnnInterface.h"
#include "System/QnnSystemInterface.h"

using Bytes = std::vector<uint8_t>;
using GetProvidersFn = Qnn_ErrorHandle_t (*)(const QnnInterface_t ***, uint32_t *);
using GetSystemProvidersFn = Qnn_ErrorHandle_t (*)(const QnnSystemInterface_t ***, uint32_t *);

constexpr uint32_t kMaxFrameBytes = 16u * 1024u * 1024u;
constexpr size_t kActionBytes = 12u * 7u * sizeof(float);
constexpr size_t kTextHiddenBytes = 21u * 768u * sizeof(float);

uint32_t read_u32_be(const uint8_t *data) {
  return (static_cast<uint32_t>(data[0]) << 24) |
         (static_cast<uint32_t>(data[1]) << 16) |
         (static_cast<uint32_t>(data[2]) << 8) | static_cast<uint32_t>(data[3]);
}

uint64_t read_u64_be(const uint8_t *data) {
  uint64_t value = 0;
  for (int index = 0; index < 8; ++index) value = (value << 8) | data[index];
  return value;
}

void append_u32_be(std::string *out, uint32_t value) {
  out->push_back(static_cast<char>((value >> 24) & 0xff));
  out->push_back(static_cast<char>((value >> 16) & 0xff));
  out->push_back(static_cast<char>((value >> 8) & 0xff));
  out->push_back(static_cast<char>(value & 0xff));
}

bool recv_all(int fd, void *buffer, size_t size) {
  auto *cursor = static_cast<uint8_t *>(buffer);
  while (size != 0) {
    const ssize_t received = recv(fd, cursor, size, MSG_WAITALL);
    if (received <= 0) return false;
    cursor += received;
    size -= static_cast<size_t>(received);
  }
  return true;
}

bool send_all(int fd, const void *buffer, size_t size) {
  const auto *cursor = static_cast<const uint8_t *>(buffer);
  while (size != 0) {
    const ssize_t sent = send(fd, cursor, size, MSG_NOSIGNAL);
    if (sent <= 0) return false;
    cursor += sent;
    size -= static_cast<size_t>(sent);
  }
  return true;
}

uint64_t now_us() {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                   std::chrono::steady_clock::now().time_since_epoch())
                                   .count());
}

size_t dtype_bytes(Qnn_DataType_t dtype) {
  switch (dtype) {
    case QNN_DATATYPE_FLOAT_32:
    case QNN_DATATYPE_INT_32:
    case QNN_DATATYPE_UINT_32:
    case QNN_DATATYPE_SFIXED_POINT_32:
    case QNN_DATATYPE_UFIXED_POINT_32:
      return 4;
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_BFLOAT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    default:
      throw std::runtime_error("unsupported QNN tensor dtype");
  }
}

uint64_t tensor_elements(const Qnn_Tensor_t &tensor) {
  const uint32_t rank = tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.rank : tensor.v1.rank;
  const uint32_t *dims =
      tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.dimensions : tensor.v1.dimensions;
  uint64_t count = 1;
  for (uint32_t index = 0; index < rank; ++index) count *= dims[index];
  return count;
}

const char *tensor_name(const Qnn_Tensor_t &tensor) {
  return tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.name : tensor.v1.name;
}

Qnn_DataType_t tensor_dtype(const Qnn_Tensor_t &tensor) {
  return tensor.version == QNN_TENSOR_VERSION_2 ? tensor.v2.dataType : tensor.v1.dataType;
}

size_t tensor_bytes(const Qnn_Tensor_t &tensor) {
  return static_cast<size_t>(tensor_elements(tensor)) * dtype_bytes(tensor_dtype(tensor));
}

void bind_tensor(Qnn_Tensor_t *tensor, Qnn_TensorType_t type, void *data, size_t size) {
  if (tensor->version == QNN_TENSOR_VERSION_2) {
    tensor->v2.type = type;
    tensor->v2.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v2.clientBuf.data = data;
    tensor->v2.clientBuf.dataSize = static_cast<uint32_t>(size);
  } else {
    tensor->v1.type = type;
    tensor->v1.memType = QNN_TENSORMEMTYPE_RAW;
    tensor->v1.clientBuf.data = data;
    tensor->v1.clientBuf.dataSize = static_cast<uint32_t>(size);
  }
}

struct TensorSpec {
  Qnn_Tensor_t tensor{};
  std::string name;
  std::vector<uint32_t> dimensions;
  std::vector<uint8_t> dynamic_dimensions;

  void copy_from(const Qnn_Tensor_t &source) {
    tensor = source;
    name = tensor_name(source) ? tensor_name(source) : "";
    const uint32_t rank = source.version == QNN_TENSOR_VERSION_2 ? source.v2.rank : source.v1.rank;
    const uint32_t *dims = source.version == QNN_TENSOR_VERSION_2 ? source.v2.dimensions
                                                                    : source.v1.dimensions;
    dimensions.assign(dims, dims + rank);
    if (source.version == QNN_TENSOR_VERSION_2 && source.v2.isDynamicDimensions) {
      dynamic_dimensions.assign(source.v2.isDynamicDimensions, source.v2.isDynamicDimensions + rank);
    }
    if (tensor.version == QNN_TENSOR_VERSION_2) {
      tensor.v2.name = name.c_str();
      tensor.v2.dimensions = dimensions.empty() ? nullptr : dimensions.data();
      tensor.v2.isDynamicDimensions = dynamic_dimensions.empty() ? nullptr : dynamic_dimensions.data();
    } else {
      tensor.v1.name = name.c_str();
      tensor.v1.dimensions = dimensions.empty() ? nullptr : dimensions.data();
    }
  }
};

struct Runtime {
  void *backend_library = nullptr;
  void *system_library = nullptr;
  const QNN_INTERFACE_VER_TYPE *provider = nullptr;
  const QNN_SYSTEM_INTERFACE_VER_TYPE *system = nullptr;
  Qnn_DeviceHandle_t device = nullptr;
  Qnn_BackendHandle_t backend = nullptr;

  ~Runtime() {
    if (backend && provider && provider->backendFree) provider->backendFree(backend);
    if (device && provider && provider->deviceFree) provider->deviceFree(device);
    if (system_library) dlclose(system_library);
    if (backend_library) dlclose(backend_library);
  }

  static void check(Qnn_ErrorHandle_t status, const std::string &operation) {
    if (status == QNN_SUCCESS) return;
    std::ostringstream message;
    message << operation << " failed: 0x" << std::hex << static_cast<unsigned>(status);
    throw std::runtime_error(message.str());
  }

  void initialize(const std::string &library_root) {
    backend_library = dlopen((library_root + "/libQnnHtp.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!backend_library) throw std::runtime_error("could not open libQnnHtp.so: " + std::string(dlerror()));
    const auto get_providers = reinterpret_cast<GetProvidersFn>(
        dlsym(backend_library, "QnnInterface_getProviders"));
    if (!get_providers) throw std::runtime_error("QnnInterface_getProviders is missing");
    const QnnInterface_t **providers = nullptr;
    uint32_t provider_count = 0;
    check(get_providers(&providers, &provider_count), "QnnInterface_getProviders");
    if (!providers || provider_count == 0) throw std::runtime_error("QNN provider list is empty");
    provider = &providers[0]->QNN_INTERFACE_VER_NAME;

    system_library = dlopen((library_root + "/libQnnSystem.so").c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!system_library) throw std::runtime_error("could not open libQnnSystem.so: " + std::string(dlerror()));
    const auto get_system_providers = reinterpret_cast<GetSystemProvidersFn>(
        dlsym(system_library, "QnnSystemInterface_getProviders"));
    if (!get_system_providers) throw std::runtime_error("QnnSystemInterface_getProviders is missing");
    const QnnSystemInterface_t **system_providers = nullptr;
    uint32_t system_count = 0;
    check(get_system_providers(&system_providers, &system_count), "QnnSystemInterface_getProviders");
    if (!system_providers || system_count == 0) throw std::runtime_error("QNN system provider list is empty");
    system = &system_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
    check(provider->deviceCreate(nullptr, nullptr, &device), "deviceCreate");
    check(provider->backendCreate(nullptr, nullptr, &backend), "backendCreate");
  }
};

class Graph {
 public:
  Graph(Runtime *runtime, std::string label, std::string path)
      : runtime_(runtime), label_(std::move(label)), path_(std::move(path)) {}

  ~Graph() {
    if (context_) runtime_->provider->contextFree(context_, nullptr);
    if (mapped_) munmap(mapped_, mapped_size_);
    if (fd_ >= 0) close(fd_);
  }

  void load() {
    fd_ = open(path_.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("could not open context " + path_);
    struct stat info {};
    if (fstat(fd_, &info) != 0 || info.st_size <= 0) throw std::runtime_error("invalid context " + path_);
    mapped_size_ = static_cast<size_t>(info.st_size);
    mapped_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
      mapped_ = nullptr;
      throw std::runtime_error("mmap failed for " + path_);
    }

    QnnSystemContext_Handle_t system_context = nullptr;
    Runtime::check(runtime_->system->systemContextCreate(&system_context), "systemContextCreate");
    const QnnSystemContext_BinaryInfo_t *binary = nullptr;
    Runtime::check(runtime_->system->systemContextGetMetaData(system_context, mapped_, mapped_size_, &binary),
                   "systemContextGetMetaData");
    QnnSystemContext_GraphInfo_t *graphs = nullptr;
    uint32_t graph_count = 0;
    if (binary->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
      graphs = binary->contextBinaryInfoV3.graphs;
      graph_count = binary->contextBinaryInfoV3.numGraphs;
    } else if (binary->version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
      graphs = binary->contextBinaryInfoV2.graphs;
      graph_count = binary->contextBinaryInfoV2.numGraphs;
    } else {
      graphs = binary->contextBinaryInfoV1.graphs;
      graph_count = binary->contextBinaryInfoV1.numGraphs;
    }
    if (!graphs || graph_count == 0) {
      runtime_->system->systemContextFree(system_context);
      throw std::runtime_error("context contains no graphs " + path_);
    }
    const auto &metadata = graphs[0].graphInfoV1;
    graph_name_ = metadata.graphName ? metadata.graphName : "";
    if (graph_name_.empty()) {
      runtime_->system->systemContextFree(system_context);
      throw std::runtime_error("context graph has no name " + path_);
    }
    for (uint32_t index = 0; index < metadata.numGraphInputs; ++index) {
      TensorSpec spec;
      spec.copy_from(metadata.graphInputs[index]);
      inputs_.push_back(std::move(spec));
    }
    for (uint32_t index = 0; index < metadata.numGraphOutputs; ++index) {
      TensorSpec spec;
      spec.copy_from(metadata.graphOutputs[index]);
      outputs_.push_back(std::move(spec));
    }
    runtime_->system->systemContextFree(system_context);
    Runtime::check(runtime_->provider->contextCreateFromBinary(runtime_->backend, runtime_->device, nullptr,
                                                                mapped_, mapped_size_, &context_, nullptr),
                   "contextCreateFromBinary " + label_);
    Runtime::check(runtime_->provider->graphRetrieve(context_, graph_name_.c_str(), &graph_),
                   "graphRetrieve " + label_);
    std::printf("loaded %-14s graph=%s inputs=%zu outputs=%zu\n", label_.c_str(), graph_name_.c_str(),
                inputs_.size(), outputs_.size());
  }

  Bytes execute(const std::map<std::string, const Bytes *> &provided, uint64_t *elapsed_us) const {
    std::vector<Qnn_Tensor_t> inputs;
    std::vector<Qnn_Tensor_t> outputs;
    std::vector<Bytes> output_buffers;
    output_buffers.reserve(outputs_.size());
    inputs.reserve(inputs_.size());
    outputs.reserve(outputs_.size());
    for (const auto &spec : inputs_) {
      const auto found = provided.find(spec.name);
      if (found == provided.end()) throw std::runtime_error(label_ + " missing input " + spec.name);
      if (found->second->size() != tensor_bytes(spec.tensor)) {
        throw std::runtime_error(label_ + " input size mismatch for " + spec.name);
      }
      Qnn_Tensor_t tensor = spec.tensor;
      bind_tensor(&tensor, QNN_TENSOR_TYPE_APP_WRITE, const_cast<uint8_t *>(found->second->data()),
                  found->second->size());
      inputs.push_back(tensor);
    }
    for (const auto &spec : outputs_) {
      output_buffers.emplace_back(tensor_bytes(spec.tensor));
      Qnn_Tensor_t tensor = spec.tensor;
      bind_tensor(&tensor, QNN_TENSOR_TYPE_APP_READ, output_buffers.back().data(), output_buffers.back().size());
      outputs.push_back(tensor);
    }
    if (outputs.size() != 1) throw std::runtime_error(label_ + " must expose one output");
    const uint64_t started = now_us();
    Runtime::check(runtime_->provider->graphExecute(graph_, inputs.data(), static_cast<uint32_t>(inputs.size()),
                                                    outputs.data(), static_cast<uint32_t>(outputs.size()), nullptr,
                                                    nullptr),
                   "graphExecute " + label_);
    *elapsed_us += now_us() - started;
    return std::move(output_buffers.front());
  }

 private:
  Runtime *runtime_;
  std::string label_;
  std::string path_;
  int fd_ = -1;
  void *mapped_ = nullptr;
  size_t mapped_size_ = 0;
  std::string graph_name_;
  Qnn_ContextHandle_t context_ = nullptr;
  Qnn_GraphHandle_t graph_ = nullptr;
  std::vector<TensorSpec> inputs_;
  std::vector<TensorSpec> outputs_;
};

struct Request {
  std::map<std::string, Bytes> arrays;
};

Request parse_request(const std::string &payload) {
  if (payload.size() < 4) throw std::runtime_error("request is truncated");
  Request request;
  const auto *data = reinterpret_cast<const uint8_t *>(payload.data());
  size_t offset = 0;
  const uint32_t count = read_u32_be(data + offset);
  offset += 4;
  if (count == 0 || count > 32) throw std::runtime_error("invalid request tensor count");
  for (uint32_t index = 0; index < count; ++index) {
    if (offset + 2 > payload.size()) throw std::runtime_error("request name is truncated");
    const uint16_t name_size = static_cast<uint16_t>((data[offset] << 8) | data[offset + 1]);
    offset += 2;
    if (name_size == 0 || offset + name_size + 8 > payload.size()) {
      throw std::runtime_error("request tensor header is invalid");
    }
    const std::string name(reinterpret_cast<const char *>(data + offset), name_size);
    offset += name_size;
    const uint64_t size = read_u64_be(data + offset);
    offset += 8;
    if (size > kMaxFrameBytes || size > payload.size() - offset) {
      throw std::runtime_error("request tensor size is invalid");
    }
    Bytes value(static_cast<size_t>(size));
    std::memcpy(value.data(), data + offset, value.size());
    offset += value.size();
    if (!request.arrays.emplace(name, std::move(value)).second) {
      throw std::runtime_error("request contains a duplicate tensor name");
    }
  }
  if (offset != payload.size()) throw std::runtime_error("request has trailing bytes");
  return request;
}

const Bytes &require(const Request &request, const std::string &name) {
  const auto found = request.arrays.find(name);
  if (found == request.arrays.end()) throw std::runtime_error("missing request tensor " + name);
  return found->second;
}

class TurboVlaServer {
 public:
  TurboVlaServer(std::string model_root, std::string runtime_root, uint16_t port)
      : model_root_(std::move(model_root)), runtime_root_(std::move(runtime_root)), port_(port) {}

  void initialize() {
    runtime_.initialize(runtime_root_ + "/lib/aarch64-oe-linux-gcc11.2");
    load_graph("dinov3", "contexts/dinov3.bin");
    load_graph("bert_l11", "contexts/bert_l11.bin");
    load_graph("bert_l14", "contexts/bert_l14.bin");
    load_graph("bert_l21", "contexts/bert_l21.bin");
    load_graph("policy_core", "contexts/policy_core.bin");
    std::printf("TurboVLA QNN service ready on 0.0.0.0:%u\n", port_);
    std::fflush(stdout);
  }

  void serve() {
    const int listener = socket(AF_INET, SOCK_STREAM, 0);
    if (listener < 0) throw std::runtime_error("socket failed");
    int reuse = 1;
    setsockopt(listener, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_ANY);
    address.sin_port = htons(port_);
    if (bind(listener, reinterpret_cast<sockaddr *>(&address), sizeof(address)) != 0 ||
        listen(listener, 1) != 0) {
      close(listener);
      throw std::runtime_error("bind/listen failed");
    }
    while (true) {
      const int client = accept(listener, nullptr, nullptr);
      if (client >= 0) {
        serve_client(client);
        close(client);
      }
    }
  }

 private:
  void load_graph(const std::string &label, const std::string &relative_path) {
    auto graph = std::make_unique<Graph>(&runtime_, label, model_root_ + "/" + relative_path);
    graph->load();
    graphs_.emplace(label, std::move(graph));
  }

  Bytes run(const std::string &label, const std::map<std::string, const Bytes *> &inputs,
            uint64_t *elapsed_us) const {
    return graphs_.at(label)->execute(inputs, elapsed_us);
  }

  Bytes infer(const Request &request, std::string *metrics) const {
    uint64_t dino_us = 0;
    uint64_t bert_us = 0;
    uint64_t core_us = 0;
    const Bytes &view0 = require(request, "pixels_view0");
    const Bytes &view1 = require(request, "pixels_view1");
    const Bytes dino0 = run("dinov3", {{"pixel_values", &view0}}, &dino_us);
    const Bytes dino1 = run("dinov3", {{"pixel_values", &view1}}, &dino_us);

    const Bytes &input_ids = require(request, "input_ids");
    const size_t text_length = input_ids.size() / sizeof(int32_t);
    if (input_ids.size() != text_length * sizeof(int32_t) ||
        (text_length != 11 && text_length != 14 && text_length != 21)) {
      throw std::runtime_error("input_ids must contain exactly 11, 14, or 21 int32 tokens");
    }
    const std::string bert_name = "bert_l" + std::to_string(text_length);
    const Bytes bert = run(bert_name,
                           {{"input_ids", &input_ids},
                            {"token_type_ids", &require(request, "token_type_ids")},
                            {"text_self_attention_mask", &require(request, "bert_attention_mask")},
                            {"position_ids", &require(request, "position_ids")}},
                           &bert_us);
    if (bert.size() != text_length * 768u * sizeof(float)) {
      throw std::runtime_error("BERT returned an unexpected hidden size");
    }
    Bytes bert_padded(kTextHiddenBytes, 0);
    std::memcpy(bert_padded.data(), bert.data(), bert.size());
    Bytes action = run("policy_core",
                       {{"vision_view0", &dino0},
                        {"vision_view1", &dino1},
                        {"bert_hidden_padded", &bert_padded},
                        {"text_key_padding_mask", &require(request, "text_key_padding_mask")},
                        {"text_self_attention_mask", &require(request, "text_self_attention_mask")},
                        {"state", &require(request, "state")}},
                       &core_us);
    if (action.size() != kActionBytes) throw std::runtime_error("policy core returned an unexpected action size");
    std::ostringstream json;
    json << "{\"dino_us\":" << dino_us << ",\"bert_us\":" << bert_us
         << ",\"policy_core_us\":" << core_us << "}";
    *metrics = json.str();
    return action;
  }

  static std::string response(uint32_t status, const Bytes &action, const std::string &metrics) {
    std::string body;
    append_u32_be(&body, status);
    append_u32_be(&body, static_cast<uint32_t>(action.size()));
    body.append(reinterpret_cast<const char *>(action.data()), action.size());
    append_u32_be(&body, static_cast<uint32_t>(metrics.size()));
    body.append(metrics);
    std::string framed;
    append_u32_be(&framed, static_cast<uint32_t>(body.size()));
    framed.append(body);
    return framed;
  }

  void serve_client(int client) const {
    while (true) {
      uint8_t size[4];
      if (!recv_all(client, size, sizeof(size))) return;
      const uint32_t frame_size = read_u32_be(size);
      if (frame_size == 0 || frame_size > kMaxFrameBytes) return;
      std::string payload(frame_size, '\0');
      if (!recv_all(client, payload.data(), payload.size())) return;
      const uint64_t started = now_us();
      Bytes action;
      std::string metrics;
      uint32_t status = 0;
      try {
        action = infer(parse_request(payload), &metrics);
        metrics.insert(metrics.size() - 1, ",\"request_us\":" + std::to_string(now_us() - started));
      } catch (const std::exception &error) {
        status = 1;
        metrics = std::string("{\"error\":\"") + error.what() + "\"}";
      }
      const std::string result = response(status, action, metrics);
      if (!send_all(client, result.data(), result.size())) return;
    }
  }

  std::string model_root_;
  std::string runtime_root_;
  uint16_t port_;
  Runtime runtime_;
  std::map<std::string, std::unique_ptr<Graph>> graphs_;
};

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <model-root> <qairt-runtime-root> [port]\n", argv[0]);
    return 2;
  }
  const std::string model_root = argv[1];
  const std::string runtime_root = argv[2];
  const uint16_t port = argc > 3 ? static_cast<uint16_t>(std::strtoul(argv[3], nullptr, 10)) : 10092;
  try {
    TurboVlaServer server(model_root, runtime_root, port);
    server.initialize();
    server.serve();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "fatal: %s\n", error.what());
    return 1;
  }
  return 0;
}
