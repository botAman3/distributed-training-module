cmake_minimum_required(VERSION 3.15)
project(federated_server)

find_package(Protobuf REQUIRED)
find_package(gRPC REQUIRED)

set(PROTO_FILE gradients.proto)

# Generate gRPC sources
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILE})

add_executable(server
    server.cpp
    ${PROTO_SRCS}
    ${PROTO_HDRS}
    ${GRPC_SRCS}
    ${GRPC_HDRS}
)

target_include_directories(server PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

target_link_libraries(server
    gRPC::grpc++
    protobuf::libprotobuf
)
