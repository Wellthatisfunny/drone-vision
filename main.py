from djitellopy import Tello
import cv2
import socket
import struct
import threading
import time
import traceback
from collections import deque

class TelloSocketServer:
    def __init__(self, host='0.0.0.0', port=8888, angle_port=5560):
        """we are defining two sockets.
        1) TCP: from main.py (Windows) ---> MonoLive.cpp (WSL) | it uses address = host, port = port for sending video stream (from the drone's camera)
        2) UDP: from MonoLive.cpp (WSL) ---> main.py (Windows)| it uses port = angle_port for sending back the angle of the detected person (if exists) relative to the drone"""
        self.host = host
        self.port = port
        self.angle_port = angle_port

        #sockets
        self.server_socket = None #TCP listening socket | waits for clients
        self.client_socket = None #TCP connection socket | the communication channel
        self.udp_socket = None #UDP socket

        self.tello = None #the Tello drone

        # control flags
        self.running = False #a control flag | True ---> system is active, threads should run

        # threads
        self.angle_thread = None #thread listens on UDP for the angle the drone should turn in order to face the person
        self.square_thread = None #thread performing the square loop (the drone flies non-stop in a square when no person detected)
        self.command_worker_thread = None #pulls flight commands from a queue and executes them on the tello drone (prevents race conditions)

        # detection state
        self.human_detected = False #boolean flag | True ---> person was detected recently
        self.last_detection_time = 0.0 #the time of last detection (i.e. what was the timestamp at last detection)
        self.detection_timeout = 5.0  #5 seconds with no detections ---> set, human_detected = False
        self.detection_lock = threading.Lock() #a lock for protecting the 3 values above

        # chase state - a person was detected, the drone is chasing | search mode - flying in a square, searching for a person
        self.chase_active = False #a flag
        self.chase_lock = threading.Lock()#a lock for protection

        # command queue implemented as deque for front-insert (priority)
        # items are tuples: (tag, func, args, kwargs)
        #deque for commands (command for the Tello drone)
        #deque for front-insert (for high priority)
        #items in deque are tuples: (tag, func, args, kwargs)
        # 1) tag - search /chase / system (land, end...)
        # 2) func - function (e.g. move_forward...)
        # 3) args - arguments for the function (e.g. 85)
        # 4) kwargs - keyword arguments for the function
        self._queue = deque()
        self.queue_lock = threading.Lock() #a lock for the queue
        self.command_worker_stop = threading.Event() #an event | not set ---> command_worker_thread still running

        # iteration control for square_search
        self.iteration_in_progress = threading.Event() #an event | set ---> a square searching loop already in progress, not set ---> safe to start a new one

    # ... detection functions ...
    def update_human_detection(self, detected: bool):
        """this function updates the self.human_detected safely"""
        with self.detection_lock: #using a lock for safety
            if detected:
                # mark that a person was seen
                self.human_detected = True
                # store the exact time of this detection
                self.last_detection_time = time.time()
            else:
                # if no detection: if too much time passed since last detection ---> set human_detected to false
                if time.time() - self.last_detection_time > self.detection_timeout:
                    self.human_detected = False

    def is_human_detected(self) -> bool:
        """this function returns current state of human_detected (and updates it if needed)"""
        #really similar to previous function
        with self.detection_lock:
            # auto-clear on timeout
            if self.human_detected and (time.time() - self.last_detection_time > self.detection_timeout):
                self.human_detected = False
                print("Human detection timed out - switching to search mode")
            return self.human_detected

    # ... sockets ...
    def start_server(self):
        """Initialize TCP server socket"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM) #starting listening TCP socket
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        print(f"Server listening on {self.host}:{self.port}")

    def start_angle_receiver(self):
        """Create UDP socket, start angle receiver and square loop threads.
           Must be called after self.running = True."""

        try:
            #start the UDP socket
            self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.udp_socket.bind(("0.0.0.0", self.angle_port))
            self.udp_socket.settimeout(1.0)
            print(f"UDP angle receiver listening on 0.0.0.0:{self.angle_port}")

            # Angle receiver thread
            self.angle_thread = threading.Thread(target=self.angle_receiver_loop, daemon=True)
            # Square search thread
            self.square_thread = threading.Thread(target=self.square_loop, daemon=True)

            self.angle_thread.start()
            self.square_thread.start()

        except Exception as e:
            print(f"Error starting angle receiver: {e}")
            traceback.print_exc()

    # ... command worker ...
    def start_command_worker(self):
        """Starting the command_worker_thread"""
        self.command_worker_stop.clear()
        self.command_worker_thread = threading.Thread(target=self._command_worker_loop, daemon=True)
        self.command_worker_thread.start()

    def _command_worker_loop(self):
        """the loop itself. this function will execute the commands in the deque """
        print("Command worker started")
        while not self.command_worker_stop.is_set():
            item = None
            with self.queue_lock:
                if self._queue:
                    item = self._queue.popleft() #get the next command
            if item is None:
                time.sleep(0.05) #prevent busy waiting
                continue

            try:
                tag, func, args, kwargs = item
                if self.tello is None:
                    print(f"Command worker: tello not connected; skipping {tag} command")
                else:
                    # run the function (with the args/kwargs)
                    func(*args, **(kwargs or {}))
            except Exception as e:
                print(f"Error executing queued command: {e}")
                traceback.print_exc()
        print("Command worker stopped")

    def enqueue_command(self, func, *args, tag='general', priority=False, **kwargs):
        """put a command in the command queue | priority=True ---> insert at front so it runs next."""
        item = (tag, func, args, kwargs)
        with self.queue_lock:
            if priority:
                self._queue.appendleft(item)
            else:
                self._queue.append(item)

    def remove_search_commands(self):
        """Remove from the command queue 'search' tagged commands to give priority to chase/system commands"""
        with self.queue_lock:
            kept = deque()
            while self._queue:
                item = self._queue.popleft()
                tag = item[0]
                if tag == 'search':
                    # drop it
                    continue
                kept.append(item) #save system/chase commands
            self._queue = kept

    def queue_is_empty(self):
        with self.queue_lock:
            return len(self._queue) == 0

    # ... angle receiver ...
    def angle_receiver_loop(self):
        print("Angle receiver thread started")
        angle_count = 0

        while self.running:
            try:
                data, addr = self.udp_socket.recvfrom(1024) #read an angle from the socket
                angle_str = data.decode().strip()

                if angle_str: #if human detected
                    # mark human detected
                    self.update_human_detection(True)

                    try:
                        angle_deg = float(angle_str) #turning the angle from string to float
                        angle_count += 1

                        if angle_count % 20 == 0: #once every 20 angles print...
                            print(f"Human detected! Target Angle #{angle_count}: {angle_deg:.2f} degrees")

                        #changing to chase mode (because human was detected)
                        with self.chase_lock:
                            self.chase_active = True

                        # remove search commands
                        self.remove_search_commands()

                        # insert chase commands with priority so they execute before system commands (if there are)
                        angle_int = int(round(abs(angle_deg)))
                        turn_threshold = 5 #if angle to turn is <5 we won't
                        small_step = 40
                        if angle_int > turn_threshold:
                            if angle_deg > 0:
                                self.enqueue_command(self.tello.rotate_clockwise, angle_int, tag='chase', priority=True) #turning clockwise for positive angles
                            else:
                                self.enqueue_command(self.tello.rotate_counter_clockwise, angle_int, tag='chase', priority=True) #turning counter-clockwise for negative angles
                            # enqueue small forward step
                            self.enqueue_command(self.tello.move_forward, small_step, tag='chase', priority=True)
                        else: #if angle is <5 we would continue straight
                            self.enqueue_command(self.tello.move_forward, small_step, tag='chase', priority=True)

                        # update last detection time
                        self.update_human_detection(True)

                    except ValueError:
                        print(f"Problem with angle format: {angle_str}")

            except socket.timeout:
                #timeout ---> no person was detected
                self.update_human_detection(False) #so, updating human_detection
                #if no recent detections ---> exit chase mode
                if not self.is_human_detected():
                    with self.chase_lock:
                        if self.chase_active:
                            self.chase_active = False
                            print("Chase ended (no recent detections)")
                continue
            except Exception as e:
                if self.running:
                    print(f"Angle receiver error: {e}")
                    traceback.print_exc()
                break

        print("Angle receiver thread stopped")

    # ... search mode (square loop) ...
    def square_loop(self):
        """function performs one step square_search iterations if no human detected. making sure only one iteration active at a time and waits for queue to drain between iterations"""
        print("Square search thread started")
        while self.running:
            #if a chase is active or human detected---> wait and do not start a new iteration
            if self.is_human_detected() or self.is_chase_active():
                time.sleep(0.2)
                continue

            #only one iteration at a time
            if self.iteration_in_progress.is_set():
                time.sleep(0.05)
                continue

            #ensure command queue is empty before starting a new search iteration
            if not self.queue_is_empty():
                #wait a short time for worker to finish with the other commands
                time.sleep(0.1)
                continue

            #start an iteration
            self.iteration_in_progress.set()
            try:
                print("No human detected - executing one search iteration")

                #inserting rotation and movement as 'search' tagged commands
                print("Search: enqueue rotate 90")
                self.enqueue_command(self.tello.rotate_clockwise, 90, tag='search', priority=False)

                time.sleep(0.05)

                print("Search: enqueue move_forward 85")
                self.enqueue_command(self.tello.move_forward, 85, tag='search', priority=False)

                #wait for queue to drain (or a person was detected)
                wait_start = time.time()
                max_wait = 6.0  # safety timeout to avoid stuck
                while time.time() - wait_start < max_wait and self.running:
                    if self.is_human_detected() or self.is_chase_active():
                        print("human detected during an iteration — won't start a new search iterations")
                        break
                    if self.queue_is_empty():
                        break
                    time.sleep(0.05)

            except Exception as e:
                print(f"error in the square loop: {e}")
                traceback.print_exc()
                time.sleep(0.5)
            finally:
                self.iteration_in_progress.clear()

        print("square search thread stopped")

    def is_chase_active(self):
        #returning if in chase mode
        with self.chase_lock:
            return self.chase_active

    # ...process frames ...
    def send_frame(self, frame):
        """send frames to clients, and returning if everything OK"""
        try:
            if not self.client_socket: #exit if no TCP client
                return False
            #the bytes (of the current frame)
            _, buffer = cv2.imencode('.jpg', frame)
            data = buffer.tobytes()
            #size (in bytes) of img
            size = struct.pack("<I", len(data))
            #sending to clients
            self.client_socket.sendall(size)
            self.client_socket.sendall(data)
            return True
        except Exception as e:
            print(f"Error sending frame: {e}")
            return False

    def run_stream(self):
        """Main streaming loop. Starts receiver threads and command worker, then streams frames."""
        # set system as running
        self.running = True

        # Start command worker
        self.start_command_worker()

        # Start angle receiver and square search
        self.start_angle_receiver()

        frame_count = 0

        try:
            while self.running:
                try:
                    if self.tello is None: #if no drone, sleep and try again (no busy waiting)
                        time.sleep(0.1)
                        continue

                    frame_read = self.tello.get_frame_read() #get current frame from tello
                    if frame_read is None: #if no frame sleep and try again
                        time.sleep(0.02)
                        continue

                    frame = frame_read.frame
                    if frame is None:
                        time.sleep(0.02)
                        continue

                    if not self.send_frame(frame): #client disconnected ---> stop streaming
                        print("Failed to send frame to client; stopping stream")
                        break

                    frame_count += 1
                    if frame_count % 30 == 0: #every 30 frames print if human was detected
                        human_status = "DETECTED" if self.is_human_detected() else "NOT DETECTED"
                        print(f"Sent {frame_count} frames - Human: {human_status}")

                    # for ~30 FPS
                    time.sleep(0.033)
                except Exception as e:
                    print(f"Stream loop inner error: {e}")
                    traceback.print_exc()
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("Stream interrupted by user")
        except Exception as e:
            print(f"Stream error (outer): {e}")
            traceback.print_exc()
        finally:
            self.cleanup()

    # ... tello ...
    def connect_tello(self):
        """connect to tello and start video. insert commands to command queue"""
        try:
            #connecting to tello
            self.tello = Tello()
            self.tello.connect()
            #printing battery
            battery = self.tello.get_battery()
            print(f"Tello connected. Battery: {battery}%")

            # start stream on Tello
            self.tello.streamon()
            print("Tello video stream enabled")

            # safe takeoff: enqueue takeoff commands (tag - system, priority - True to run soon)
            self.enqueue_command(self.safe_takeoff, tag='system', priority=True)

        except Exception as e:
            print(f"Failed to connect to Tello: {e}")
            traceback.print_exc()
            # keep tello reference for possible cleanup but do not raise

    def safe_takeoff(self):
        try:
            #check connection
            if self.tello is None:
                print("Safe takeoff: no tello instance")
                return
            print("Taking off...")

            # attempt takeoff
            self.tello.takeoff()
            print("Takeoff OK")
        except Exception as e:
            print(f"Takeoff failed: {e}")
            traceback.print_exc()

    # ... connecting client ...
    def wait_for_client(self):
        print("wwaiting for client connection...")
        try:
            self.client_socket, addr = self.server_socket.accept()
            print(f"Client connected from {addr}")
        except Exception as e:
            print(f"Error accepting client: {e}")
            traceback.print_exc()

    # ... cleanup ...
    def cleanup(self):
        """clean up. safe to call multiple times."""
        print("Cleaning up server...")
        # stop running loops
        self.running = False

        #close UDP socket
        try:
            if self.udp_socket:
                self.udp_socket.close()
                self.udp_socket = None
        except Exception:
            pass

        #stop command worker
        try:
            self.command_worker_stop.set()
            if self.command_worker_thread and self.command_worker_thread.is_alive():
                self.command_worker_thread.join(timeout=2)
            #clear the command queue
            with self.queue_lock:
                self._queue.clear()
        except Exception:
            pass

        #stop angle and square threads
        try:
            if self.angle_thread and self.angle_thread.is_alive():
                self.angle_thread.join(timeout=2)
            if self.square_thread and self.square_thread.is_alive():
                self.square_thread.join(timeout=2)
        except Exception:
            pass

        #streamoff and land
        try:
            if self.tello: #if still connected to tello ---> close stream + land safely
                try:
                    #close video stream
                    self.tello.streamoff()
                except Exception:
                    pass
                try:
                    #lands
                    self.tello.land()
                except Exception:
                    pass
                try:
                    self.tello.end()  #call tello own cleanup
                except Exception:
                    pass
                self.tello = None
        except Exception:
            pass

        #close client and server sockets
        try:
            #close client_socket
            if self.client_socket:
                try:
                    self.client_socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.client_socket.close()
                self.client_socket = None
        except Exception:
            pass

        try:
            if self.server_socket:
                try:
                    self.server_socket.shutdown(socket.SHUT_RDWR)
                except Exception:
                    pass
                self.server_socket.close()
                self.server_socket = None
        except Exception:
            pass

        print("Server cleaned up")

# ... main ...
#yep, final function...
def main():
    server = TelloSocketServer() #create an instance

    try:
        server.start_server() #start the
        server.connect_tello() #start TCP socket
        server.wait_for_client() #wait for clients

        # start streaming (also starts threads + command worker)
        server.run_stream()
    except Exception as e:
        print(f"Server fatal error: {e}")
        traceback.print_exc()
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()

#YESSSS. We did it! Now to MonoLive.cpp