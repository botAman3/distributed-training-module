import socket
import threading
import pickle
import struct
import torch
import logging
from typing import Dict, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HOST = 'localhost'
PORT = 50050
NUM_CLIENTS = 1
NUM_EPOCHS = 5

class ServerState:
    def __init__(self):
        self.client_sockets: Dict[str, socket.socket] = {}
        self.client_gradients: Dict[str, Dict] = {}
        self.lock = threading.Lock()
        self.current_epoch = 0
        self.cv = threading.Condition()

def send_msg(sock: socket.socket, msg: Dict[str, Any]) -> bool:
    """Send a message with length prefix"""
    try:
        data = pickle.dumps(msg)
        msg_len = len(data)
        header = struct.pack('>I', msg_len)
        sock.sendall(header + data)
        return True
    except (pickle.PickleError, struct.error, socket.error) as e:
        logger.error(f"Failed to send message: {e}")
        return False

def recv_msg(sock: socket.socket) -> Optional[Dict[str, Any]]:
    """Receive a length-prefixed message"""
    try:
        # Read message length (4 bytes)
        header = sock.recv(4)
        if len(header) != 4:
            return None
            
        msg_len = struct.unpack('>I', header)[0]

        # Read message data
        chunks = []
        bytes_recd = 0
        while bytes_recd < msg_len:
            chunk = sock.recv(min(msg_len - bytes_recd, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            bytes_recd += len(chunk)

        if bytes_recd != msg_len:
            logger.error("Incomplete message received")
            return None

        data = b''.join(chunks)
        return pickle.loads(data)
    except (struct.error, socket.error, pickle.PickleError) as e:
        logger.error(f"Failed to receive message: {e}")
        return None

def aggregate_gradients(gradients: Dict[str, Dict]) -> Dict[str, Any]:
    """Aggregate gradients from all clients"""
    logger.info("🔁 Aggregating gradients...")
    agg_result = {}

    if not gradients:
        return agg_result

    # Assume all clients sent same keys
    first_client = next(iter(gradients))
    for name in gradients[first_client]:
        stacked = [torch.tensor(gradients[c][name]["values"]).view(gradients[c][name]["shape"]) 
                 for c in gradients]
        avg = torch.stack(stacked).mean(dim=0)
        agg_result[name] = {
            "values": avg.view(-1).tolist(),
            "shape": list(avg.shape)
        }
    return agg_result

def handle_client(conn: socket.socket, addr: tuple, client_id: str, state: ServerState):
    logger.info(f"[+] {client_id} connected from {addr}")
    
    try:
        # Send initial training start message
        if not send_msg(conn, {"type": "start_training", "epoch": state.current_epoch}):
            return

        while state.current_epoch < NUM_EPOCHS:
            msg = recv_msg(conn)
            if msg is None:
                break

            if msg.get("type") == "gradients":
                with state.lock:
                    state.client_gradients[client_id] = msg["data"]
                    logger.info(f"📥 Received gradients from {client_id}")

                    if len(state.client_gradients) == NUM_CLIENTS:
                        aggregated = aggregate_gradients(state.client_gradients)

                        # Send to all clients
                        for cid, csock in state.client_sockets.items():
                            if not send_msg(csock, {
                                "type": "updated_gradients",
                                "epoch": state.current_epoch,
                                "data": aggregated
                            }):
                                logger.error(f"Failed to send to {cid}")
                                continue

                        state.client_gradients.clear()
                        state.current_epoch += 1
                        logger.info(f"✅ Epoch {state.current_epoch - 1} complete")

                        # Notify all waiting threads
                        with state.cv:
                            state.cv.notify_all()

            elif msg.get("type") == "ready":
                # Wait for next epoch
                with state.cv:
                    while (state.current_epoch < NUM_EPOCHS and 
                           len(state.client_gradients) < NUM_CLIENTS):
                        state.cv.wait()

    except Exception as e:
        logger.error(f"❌ Error with {client_id}: {str(e)}")
    finally:
        with state.lock:
            if client_id in state.client_sockets:
                del state.client_sockets[client_id]
            if client_id in state.client_gradients:
                del state.client_gradients[client_id]
        logger.info(f"[-] {client_id} disconnected")
        conn.close()

def main():
    state = ServerState()
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()

        logger.info(f"🟢 Server listening on {HOST}:{PORT}")

        try:
            while len(state.client_sockets) < NUM_CLIENTS:
                conn, addr = server.accept()
                client_id = f"client_{len(state.client_sockets)+1}"
                
                with state.lock:
                    state.client_sockets[client_id] = conn

                thread = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, client_id, state),
                    daemon=True
                )
                thread.start()

            logger.info("🎯 All clients connected. Waiting for training to complete...")

            # Wait for training to finish
            while state.current_epoch < NUM_EPOCHS:
                time.sleep(1)

            logger.info("🏁 Training complete. Shutting down.")

        except KeyboardInterrupt:
            logger.info("Server shutting down...")
        finally:
            # Close all client connections
            with state.lock:
                for sock in state.client_sockets.values():
                    sock.close()
            server.close()

if __name__ == "__main__":
    main()
