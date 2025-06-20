import grpc
import unary.unary_pb2_grpc as pb2_grpc
import unary.unary_pb2 as pb2

class Client(object):
    def __init__(self):
        self.host = "localhost"
        self.server_port = "8082"
        try:
         self.channel =  grpc.insecure_channel('{},{}'.format(self.host,self.server_port))
         self.stub = pub2_grpc.UnaryStub(self.channel)
        except Exception as e:
            print(f"Exception occoured while defining the client with message :{e}")
            
        
    def get_url(self,clientMessage):
        message = pb2.Message(message=clientMessage)
        print(f'Message')
        get_data = self.stub.GetServerResponse(message)
        
        
    
if __name__ == "__main__":
    client = UnaryClient()
    result = client.get_url(message="Hello Server Please send the batch data set")
    print(result)
        