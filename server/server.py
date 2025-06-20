import grpc
from concurrent import futures
from collections import defaultdict
import gradients_pb2
import gradients_pb2_grpc
import queue
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_CLIENTS = 1
NUM_EPOCHS = 5

class ClientSession:
    def __init__(self):
        self.q = queue.Queue()
        self.active = True

class GradientServer(gradients_pb2_grpc.GradientServiceServicer):
    def __init__(self):
        self.clients = {}
        self.grad_store = {}
        self.epoch = 0
        self.lock = threading.Lock()
        self.cv = threading.Condition()

    def FederatedTraining(self, request_iterator, context):
        try:
            first_msg = next(request_iterator)
            client_id = first_msg.join.client_id
            logger.info(f"[+] {client_id} joined")
            
           
            with self.lock:
                session = ClientSession()
                self.clients[client_id] = session

            with self.lock:
             if len(self.clients) == NUM_CLIENTS and self.epoch == 0:
                threading.Thread(target=self.training_loop, daemon=True).start()

            def recv():
                try:
                    for msg in request_iterator:
                        if msg.HasField("gradients"):
                            with self.lock:
                                self.grad_store[msg.gradients.client_id] = msg.gradients
                                logger.info(f"[✓] Got gradients from {msg.gradients.client_id}")
                                self.cv.notify_all() 
                except Exception as e:
                    logger.error(f"Error in receive thread: {str(e)}")
                finally:
                    with self.lock:
                        session.active = False
                        self.cv.notify_all()  

            recv_thread = threading.Thread(target=recv, daemon=True)
            recv_thread.start()

            while True:
                if not session.active:
                    break
                try:
                    msg = session.q.get(timeout=1)
                    yield msg
                except queue.Empty:
                    if not session.active:
                        break
                    continue

        except Exception as e:
            logger.error(f"Error in client stream: {str(e)}")
        finally:
            with self.lock:
                if client_id in self.clients:
                    del self.clients[client_id]
                    logger.info(f"[-] {client_id} disconnected")

    def training_loop(self):
        try:
            for epoch in range(NUM_EPOCHS):
                logger.info(f"\n🚀 Epoch {epoch} starting")
                
                # Signal all clients to start
                with self.lock:
                    self.grad_store.clear()
                    for client_id, session in self.clients.items():
                        msg = gradients_pb2.ServerMessage(instruction="start_training")
                        session.q.put(msg)

                # Wait for all gradients
                with self.cv:
                    while len(self.grad_store) < len(self.clients):
                        self.cv.wait()
                        if any(not session.active for session in self.clients.values()):
                            logger.error("Some client disconnected, aborting training")
                            return

                # Aggregate
                agg = defaultdict(list)
                for packet in self.grad_store.values():
                    for grad in packet.gradients:
                        agg[grad.name].append((grad.values, grad.shape))

                # Mean aggregation
                final = []
                for name, vals in agg.items():
                    avg = [sum(x) / len(vals) for x in zip(*[v[0] for v in vals])]
                    final.append(gradients_pb2.Gradient(name=name, values=avg, shape=vals[0][1]))

                # Send to all clients
                with self.lock:
                    for client_id, session in self.clients.items():
                        if session.active:
                            msg = gradients_pb2.ServerMessage(
                                updated_gradients=gradients_pb2.GradientPacket(
                                    client_id=client_id, gradients=final, epoch=epoch
                                )
                            )
                            session.q.put(msg)

                self.epoch += 1
                logger.info(f"✅ Epoch {epoch} complete")

        except Exception as e:
            logger.error(f"Error in training loop: {str(e)}")

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    gradients_pb2_grpc.add_GradientServiceServicer_to_server(GradientServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logger.info("🟢 Server running at 50051")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()