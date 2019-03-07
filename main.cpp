// by Haoze Sun 2019.3.6
// a demo for qa-select using tf serving.

#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include "grpc++/grpc++.h"
#include "tensorflow_serving/apis/prediction_service.grpc.pb.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;

using tensorflow::serving::PredictRequest;
using tensorflow::serving::PredictResponse;
using tensorflow::serving::PredictionService;

typedef google::protobuf::Map<std::string, tensorflow::TensorProto> FFDict;  // feed & fetch dict

// override of c++ types into tensorflow protobuf type
// 1. string/boolean/int32/float32
// 2. data in row major
// github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor.proto
// github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/tensor_shape.proto
tensorflow::TensorProto transFormat(const char * arr, int & arr_size) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_STRING);
    proto.add_string_val(arr, arr_size);
    proto.mutable_tensor_shape()->add_dim()->set_size(1);
    return proto;
}

tensorflow::TensorProto transFormat(const float arr) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_FLOAT);
    proto.add_float_val(arr);
    //scalar has no shape.
    return proto;
}

tensorflow::TensorProto transFormat(const bool arr) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_BOOL);
    proto.add_bool_val(arr);
    return proto;
}

tensorflow::TensorProto transFormat(const std::vector<std::string> & arr, std::vector<int> & shapes) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_STRING);
    for (int i = 0; i < arr.size(); ++i) {
        proto.add_string_val(arr[i]);
    }
    for (int i = 0; i < shapes.size(); ++i) {
        proto.mutable_tensor_shape()->add_dim()->set_size(shapes[i]);
    }
    return proto;
}

tensorflow::TensorProto transFormat(const std::vector<bool> & arr, std::vector<int> & shapes) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_BOOL);
    for (int i = 0; i < arr.size(); ++i) {
        proto.add_bool_val(arr[i]);
    }
    for (int i = 0; i < shapes.size(); ++i) {
        proto.mutable_tensor_shape()->add_dim()->set_size(shapes[i]);
    }
    return proto;
}

tensorflow::TensorProto transFormat(const std::vector<int> & arr, std::vector<int> & shapes) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_INT32);
    for (int i = 0; i < arr.size(); ++i) {
        proto.add_int_val(arr[i]);
    }
    for (int i = 0; i < shapes.size(); ++i) {
        proto.mutable_tensor_shape()->add_dim()->set_size(shapes[i]);
    }
    return proto;
}

tensorflow::TensorProto transFormat(const std::vector<float> arr, std::vector<int> & shapes) {
    tensorflow::TensorProto proto;
    proto.set_dtype(tensorflow::DataType::DT_FLOAT);
    for (int i = 0; i < arr.size(); ++i) {
        ;
        proto.add_float_val(arr[i]);
    }
    for (int i = 0; i < shapes.size(); ++i) {
        proto.mutable_tensor_shape()->add_dim()->set_size(shapes[i]);
    }
    return proto;
}

// extract data from TensorProto
bool transFormat(const tensorflow::TensorProto & message, std::vector<float> & v, int check_length=-1) {
    const google::protobuf::Descriptor* descriptor = message.GetDescriptor();
    const google::protobuf::Reflection* reflection = message.GetReflection();
    // first find the field with name
    const google::protobuf::FieldDescriptor* field = descriptor->FindFieldByName("float_val");

    // check length of the filed
    int num_elements = reflection->FieldSize(message, field);
    if (!(check_length < 0 || check_length == num_elements)) {
        return false;
    }
    float tmp;
    for (int i = 0; i < num_elements; ++i) {
        tmp = reflection->GetRepeatedFloat(message, field, i);
        v.push_back(tmp);
    }
    return true;
}

class ServingClient {
public:
    ServingClient(std::string port, std::string model_name, std::string model_signature_name);

    std::string callPredict(FFDict& feed_dict, FFDict& fetch_dict);

private:
    std::unique_ptr<PredictionService::Stub> stub_;
    std::string model_name_;
    std::string model_signature_name_;    //serving_default
};

ServingClient::ServingClient(std::string port, std::string model_name, std::string model_signature_name) {
    // setup grpc connection
    std::shared_ptr<Channel> channel = grpc::CreateChannel(port, grpc::InsecureChannelCredentials());
    // setup service
    this->stub_ = PredictionService::NewStub(channel);
    this->model_name_ = model_name;
    this->model_signature_name_ = model_signature_name;
}

std::string ServingClient::callPredict(FFDict& feed_dict, FFDict& fetch_dict) {
    PredictRequest predictRequest;
    PredictResponse response;
    ClientContext context;

    predictRequest.mutable_model_spec()->set_name(this->model_name_);
    predictRequest.mutable_model_spec()->set_signature_name(this->model_signature_name_);  // "serving_default"  as default
    FFDict & inputs = *predictRequest.mutable_inputs();
    // loop through and copy from feed dict
    for (FFDict::iterator it = feed_dict.begin(); it != feed_dict.end(); it++) {
        inputs[it->first] = it->second;
    }

    Status status = this->stub_->Predict(&context, predictRequest, &response);
    if (status.ok()) {
        // copy results form response
        FFDict & outputs = *response.mutable_outputs();
        for (FFDict::iterator it = outputs.begin(); it != outputs.end(); it++) {
            fetch_dict[std::string(it->first)] = it->second;
        }
        return "success";
    }
    else {
        return status.error_message();
    }
}


int main() {
    // lookup vocab table, words --> indices
    // test data
    int q[] = { 363, 734, 4297, 7596, 134, 26151, 848, 68, 277, 9712 };
    int p[] = { 26151, 2, 11, 545, 11, 415, 1445, 226, 11, 1328, 361, 11, 1366, 2958, 857, 1, 226, 11, 1084, 1, 228, 172, 870,
        857, 12, 1445, 27, 172, 114, 1328, 985, 26151, 361, 10, 247, 8106, 1, 226, 11, 1084, 26151, 1, 346, 11, 1414, 1250, 488, 2 };
    std::vector<int> q_v(q, q + 10);
    std::vector<int> p_v(p, p + 48);
    int qs[] = { 1, 10 };
    int ps[] = { 1, 1, 48 };
    std::vector<int> q_shapes(qs, qs + 2);
    std::vector<int> p_shapes(ps, ps + 3);
    int qm[] = { 10 };
    int pm[] = { 48 };
    std::vector<int> qm_v(qm, qm + 1);
    std::vector<int> pm_v(pm, pm + 1);
    int qms[] = { 1};
    int pms[] = { 1, 1 };
    std::vector<int> qm_shapes(qms, qms + 1);
    std::vector<int> pm_shapes(pms, pms + 2);

    // start test
    ServingClient sv("localhost:7500", "qa", "serving_default");
    FFDict feed_dict, fetch_dict;
    std::vector<float> score;
    feed_dict["keep_prob"] = transFormat(1.0f);
    feed_dict["is_training"] = transFormat(false);
    feed_dict["q"] = transFormat(q_v, q_shapes);
    feed_dict["p"] = transFormat(p_v, p_shapes);
    feed_dict["qm"] = transFormat(qm_v, qm_shapes);
    feed_dict["pm"] = transFormat(pm_v, pm_shapes);

    std::cout<<sv.callPredict(feed_dict, fetch_dict)<<std::endl;
    transFormat(fetch_dict["score"], score, 1);
    std::cout << "Result: "<<score[0] << std::endl;

    return 0;

}
