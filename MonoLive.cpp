//For understanding this cpp file better, you should read the main.py file before
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <mutex>
#include <iomanip>
#include <sstream>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
//// #include "System.h"
#include "pose_tx.h"

// Default paths
#define DEFAULT_SOCKET_HOST "172.23.64.1"  // IP for address for the Windows host
#define DEFAULT_SOCKET_PORT 8888 // Port for TCP socket
#define DEFAULT_CAM        "../camera.yaml" // yaml file we will use
#define USER_DIR           "/home/itamar" // user directory
#define DEV_DIR USER_DIR "/Dev" // Dev directory (/home/itamar/Dev)
#define MOBILE_NET_DIR DEV_DIR "/MobileNet-SSD" // Directory to "full body detection" model
#define DEFAULT_PROTOTXT   MOBILE_NET_DIR "/deploy.prototxt" // Path to file that describes structure of the MobileNet model
#define DEFAULT_CAFFEMODEL MOBILE_NET_DIR "/mobilenet_iter_73000.caffemodel" // Path to the trained model weights

// width and height of frames (from drone, arrives in the TCP socket)
#define DESIRED_W         600
#define DESIRED_H         350
// fps frames comes in
#define DESIRED_FPS       30
//let's us rotate imgs for display (at the end we put on 0, as you can see)
#define ROTATE_DEG_DISPLAY 0
////ORB-SLAM
//#define POSE_TX_PORT      5556
//#define VECT_TX_PORT      5558
//Port for angles (the angles the drone should turn)
#define ANGLE_TX_PORT     5560

// Video recording settings
#define RECORD_FPS        15 //fps
#define RECORD_CODEC      cv::VideoWriter::fourcc('X','V','I','D')  // XVID compression of the video
#define RECORDINGS_DIR    USER_DIR "/recordings/" //path to file

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>
#include <cstring>
#include <string>

namespace chrono = std::chrono;

using std::string;
using std::thread;
using std::mutex;
using std::vector
using std::cout;
using std::endl;
using std::this_thread::sleep_for;
using cerr;
using cv::Mat;


//struct used for sending data using the UDP socket
struct UdpSend {

    int s{ -1 }; // the UDP file descriptor, initialized to -1 (invalid socket)
    sockaddr_in to{}; // holds the destination address (IP+Port)

    bool init(const char* ip, int port) {
        //creating the socket
        s = socket(AF_INET, SOCK_DGRAM, 0); // creates a socket descriptor (SOCK_DGRAM ---> UDP socket)
        if (s < 0)return false; // if error in previous line ---> return false
        //defining the address
        to.sin_family = AF_INET; 
        to.sin_port = htons(port); 
        to.sin_addr.s_addr = inet_addr(ip);
        return true; 
    }

    void sendMessage(const string& line) {
        //sends a text message to the 'to' address (using UDP socket)
        if (s >= 0) 
            sendto(s, line.c_str(), line.size(), 0, (sockaddr*)&to, sizeof(to)); }
};

//video recording class
class VideoRecorder {
private:
    cv::VideoWriter writer; // OpenCV object for save the video frames
    string current_filename; //the file name
    bool is_recording; // flag | true ---> we are recording
    chrono::steady_clock::time_point recording_start; // holds timestamp of when we started recording

public:
    VideoRecorder() : is_recording(false) {
        // create recordings directory if it doesn't exist
        system(("mkdir -p " + string(RECORDINGS_DIR)).c_str());
    }

    string generateFilename() {
        // generating a new file name
        static int counter = 0;  // static int (for different file names)
        ++counter;
        std::ostringstream oss;
        oss << RECORDINGS_DIR << "human_detection_" << counter << ".avi";
        return oss.str();
    }

    void startRecording(int width, int height) {
        //calling this function to start recording
        if (!is_recording) { //if not recording:
            // open a new file + give the file a name
            current_filename = generateFilename();
            writer.open(current_filename, RECORD_CODEC, RECORD_FPS,
                cv::Size(width, height), true);

            if (writer.isOpened()) {
                // if file succesfully opend:
                // 1) mark that recording started
                is_recording = true;
                // 2) save timestamp
                recording_start = chrono::steady_clock::now();
                // 4) print "started recording"
                cout << "Started recording: " << current_filename << endl;
            }
            else {
                cerr << "Failed to start recording: " << current_filename << endl;
            }
        }
    }

    void stopRecording() {
        //calling to stop recording
        if (is_recording) { // if actually recording
            writer.release(); // release the writer
            is_recording = false; // mark we stoped recording

            //calculating duration of video
            auto duration = chrono::duration_cast<chrono::seconds>(
                chrono::steady_clock::now() - recording_start).count();
        }
    }

    void addFrame(const Mat& frame) {
        //adding a frame to the file
        if (is_recording && writer.isOpened()) { //if recording (and writer is working) ---> add a frame (and update the counter)
            writer.write(frame);
        }
    }

    bool isRecording() const {
        return is_recording;
    }
};


//small helpers
static inline float nodeReal(const cv::FileNode& n) { 
    //returning value (safely) from a yaml file
    return n.empty() ? 0.f : (float)n.real(); 
}

static Mat create_callib_mat(const string& yaml, int Wf, int Hf, float& fx_out) {
    //creating intrinsic matrix K
    //[fx   0   cx]
    //[0    fy  cy]
    //[0    0    1]
    cv::FileStorage fs(yaml, cv::FileStorage::READ); //opening the file
    //read params safely
    float fx = nodeReal(fs["Camera.fx"]); if (!fx) fx = nodeReal(fs["Camera1.fx"]); if (!fx) fx = nodeReal(fs["Camera0.fx"]);
    float fy = nodeReal(fs["Camera.fy"]); if (!fy) fy = nodeReal(fs["Camera1.fy"]); if (!fy) fy = nodeReal(fs["Camera0.fy"]);
    float cx = nodeReal(fs["Camera.cx"]); if (!cx) cx = nodeReal(fs["Camera1.cx"]); if (!cx) cx = nodeReal(fs["Camera0.cx"]);
    float cy = nodeReal(fs["Camera.cy"]); if (!cy) cy = nodeReal(fs["Camera1.cy"]); if (!cy) cy = nodeReal(fs["Camera0.cy"]);
    int yw = (int)nodeReal(fs["Camera.width"]); if (!yw) yw = (int)nodeReal(fs["Camera1.width"]);
    int yh = (int)nodeReal(fs["Camera.height"]); if (!yh) yh = (int)nodeReal(fs["Camera1.height"]);
    fs.release();
    if ((!fx || !fy)) { float f = (float)std::max(Wf, Hf); fx = fy = f; cx = Wf * 0.5f; cy = Hf * 0.5f; }
    if (yw > 0 && yh > 0 && (yw != Wf || yh != Hf)) { float sx = (float)Wf / yw, sy = (float)Hf / yh; fx *= sx; fy *= sy; cx *= sx; cy *= sy; }
    fx_out = fx;
    //returning the matrix
    return (Mat_<float>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
}


//TCP socket class
class TelloSocketClient {
private:
    int sock; //the TCP socket descriptor
    struct sockaddr_in server_addr; //the server addres (IP+Port)
    string host; //IP of host
    int port; //Port we will use
    bool connected; //a flag | True ---> connected
    thread th; //thread (receiving frmaes from the socket)
    mutex m; //a lock for protection
    std::atomic<bool> run{ false }; //a flag (safe) for the thread (tells if it should keep working)
    Mat latest; //the last frame

public:
    //const'
    TelloSocketClient(const string& host = "localhost", int port = 8888)
        : host(host), port(port), connected(false) {
        sock = 0;
    }
    //dest'
    ~TelloSocketClient() {
        disconnect();
    }

    //connecting to the socket
    bool connect_to_server() {
        //returning True if successful, else False

        //creating the socet
        if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) { //<0 for catch if it fails..
            cerr << "socket creation error" << endl;
            return false;
        }

        //fill in the server address
        server_addr.sin_family = AF_INET; //telling the OS we are using IPv4 addresses
        server_addr.sin_port = htons(port);

        //converting IPv4 address from text to binary form
        if (inet_pton(AF_INET, host.c_str(), &server_addr.sin_addr) <= 0) {
            cerr << "bad address" << endl;
            return false;
        }

        //connecting the socket
        if (connect(sock, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            cerr << "connection failed" << endl;
            return false;
        }

        connected = true;//updating the flag to True
        cout << "Connected at " << host << ":" << port << endl; //printing we are succefully connected

        //sstart the frame receiving thread
        run = true;
        th = thread([this]() {
            while (run && connected) {
                Mat frame = receive_frame();
                if (!frame.empty()) {
                    std::lock_guard<mutex> lk(m);
                    latest = frame.clone();
                }
                else {
                    sleep_for(chrono::milliseconds(10));
                }
            }
            });

        return true;
    }

    //read (and return) 'size' bytes from the socket
    vector<uchar> receive_data(size_t size) {
        vector<uchar> buffer(size); //creating a buffer for the data we read
        size_t bytes_received = 0; //keep track of how many btes we already read (set to 0 at start)

        //while server connected and we didn't read enough bytes:
        while (bytes_received < size && connected) {
            //try to read the amount of bytes left (and save it at the end of the buffer)
            ssize_t result = recv(sock, buffer.data() + bytes_received, size - bytes_received, 0);

            if (result <= 0) { //if did not succeed
                cerr << "Failed to receive data, disconnecting" << endl;
                connected = false; //mark the client as disconnected
                return vector<uchar>(); //return an empty vector
            }
            bytes_received += result; //if we read more bytes ---> update counter
        }

        return buffer; //return the buffer
    }

    //read next frame
    Mat receive_frame() {
        try {
            if (!connected) return Mat(); //if not connected ---> return empty matrix

            // read a 4-bytes (uint32_t) header. the header tells us how big is the jpeg (the frame) 
            vector<uchar> size_buffer = receive_data(sizeof(uint32_t));
            if (size_buffer.empty()) return Mat(); //if buffer is empty ---> connection error ---> exit (return empty matrix)

            //cast the 4-bytes to unsigned int (32-bit)
            uint32_t frame_size = *reinterpret_cast<uint32_t*>(size_buffer.data());

            //checking frames size are normal (some sort of a sanity check)
            if (frame_size == 0 || frame_size > 10000000) { // if not ---> exit
                cerr << "Invalid frame size: " << frame_size << endl;
                return Mat();
            }

            // read next frame (the jpeg)
            vector<uchar> frame_buffer = receive_data(frame_size);
            if (frame_buffer.empty()) return Mat(); //if empty ---> exit

            //decode the jpeg into an openCV matrix
            Mat frame = cv::imdecode(frame_buffer, cv::IMREAD_COLOR);

            // resize if need to match the wanted dimensions
            if (!frame.empty() && (frame.cols != DESIRED_W || frame.rows != DESIRED_H)) {
                Mat resized; //creating a new matrix instace
                cv::resize(frame, resized, cv::Size(DESIRED_W, DESIRED_H));
                return resized;
            }

            return frame; //returning the cv Matrix

        }
        catch (const std::exception& e) {
            cerr << "Error receiving frame: " << e.what() << endl;
            return Mat();
        }
    }

    //getter for the last frame (safe)
    bool get(Mat& out) {
        std::lock_guard<mutex> lk(m);
        if (latest.empty()) return false; //if the last frame is empty ---> exit, return False
        out = latest.clone(); //returning a copy!
        return true;
    }
    //disconnecting from the socket
    void disconnect() {
        if (connected) {
            //update flags
            run = false;
            connected = false;
            if (th.joinable())
                th.join(); //stopping the fram-reading thread
            close(sock); //close the coket
            cout << "Disconnected from server" << endl;
        }
    }

    //getter for connected
    bool isConnected() const { return connected; }
};


int main(int argc, char** argv) {
    string host = DEFAULT_SOCKET_HOST; //defining the host (default windows host)
    int socket_port = DEFAULT_SOCKET_PORT; //defining the port (for TCP socket)
    string cam = DEFAULT_CAM //defining the camera (the yaml file)

    //giving an option for setting the parameters from outside (from line command)
    if (argc >= 2) host = argv[1];
    if (argc >= 3) socket_port = std::atoi(argv[2]);
    if (argc >= 4) cam = argv[3];

    //print we connecting
    cout << "Connecting to socket server at " << host << ":" << socket_port << endl;


    //creating the tello (TCP) socket (and try to connect)
    TelloSocketClient frameGrabber(host, socket_port);
    if (!frameGrabber.connect_to_server()) {
        cerr << "ERROR: cannot connect to socket server at " << host << ":" << socket_port << "\n";
        return -1;
    }

    ////2 sockets for orb slam (pose, vector)
    //UdpSend poseSend; poseSend.init("127.0.0.1", POSE_TX_PORT); 
    //UdpSend vecSend;  vecSend.init("172.23.64.1", VECT_TX_PORT);

    UdpSend angleSend; angleSend.init("172.23.64.1", ANGLE_TX_PORT);  // UDP socket for angle

    //loading the pre-trained object detection model from files
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(DEFAULT_PROTOTXT, DEFAULT_CAFFEMODEL);
    //tell opencv to use its built in code to run the model
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //run the model on cpu, not on GPU cause we don't have one :((
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    // defining 2 matrixes 
    // 1) callibMat - the intrinsic matrix (callibration matrix)
    // 2) invMat - the inverse of callibMat
    Mat callibMat, invMat;
    //using only the last frame (for detection) can be problematic. because it can change a lot.
    //so we will use EMA - a way for us to smooth it
    //let's say x's are 100 then 102 then 150 then 104. we will get:
    // Frame 1) 100 (because smooth_x was -1 before)
    // Frmae 2) 100*0.7 (ALPHA) + 102*0.3 (1-ALPHA) = 100.6
    // Frame 3) 100.6*0.7 + 150 *0.3 = 115.42
    // Frame 4) 115.42*0.7 + 104*0.3 = 111.994
    float smooth_x = -1.f, smooth_y = -1.f;
    const float ALPHA = 0.7f;
    float fx_cam = 0.f;  // the fx value from the yaml file (the distance between the camera�s lens and the sensor plane)

    // Video recording
    VideoRecorder recorder; //a new recorder instance
    bool human_detected_prev = false; //a flag | True ---> a person was detected in last frame
    int frames_without_human = 0; //amount of frames without a person
    const int STOP_RECORDING_DELAY = 90; // stop recording after 3 seconds (90 frames at 30fps) without human

    cout << "Video recordings will be saved to: " << RECORDINGS_DIR << endl;
    cout << "Starting main processing loop..." << endl;
    //while TCP socket is connected (while video stream)
    while (frameGrabber.isConnected()) {
        Mat raw_frame;
        if (!frameGrabber.get(raw_frame)) { //if no raw ---> sleep for 0.05 (no busy waiting) ---> try again
            sleep_for(chrono::milliseconds(5));
            continue;
        }


        //on the first frame this 'if' will happen
        //it will create the callibMat, anc calculate it's inverse
        if (callibMat.empty()) {
            callibMat = create_callib_mat(cam, raw_frame.cols, raw_frame.rows, fx_cam);
            invMat = callibMat.inv();
        }

        //the matrix we will display (copy of 'raw_frame')
        Mat disp = raw_frame;

        cv::Point target(-1, -1); //the cordination of person (if one was detected)
        bool human_detected_current = false; //flag | True ---> humman was detected

        //take current frame, put it in the detection model ---> matrix called 'detections'
        Mat blob = cv::dnn::blobFromImage(disp, 0.007843, cv::Size(300, 300), cv::Scalar(127.5, 127.5, 127.5), false);
        net.setInput(blob);
        Mat detections = net.forward();

        int W = disp.cols, H = disp.rows; //grab width and height (in frames) of disp
        //reshaping, and making the detections easier to work with
        //each row represent different detection
        Mat detectionMat(detections.size[2], detections.size[3], CV_32F, detections.ptr<float>());

        //going over all detections (all rows)
        for (int i = 0; i < detectionMat.rows; ++i) {
            float confidence = detectionMat.at<float>(i, 2); //3rd colomn is the confidence colomn - how confidente the model in the detection it made
            int class_id = static_cast<int>(detectionMat.at<float>(i, 1)); //the 2nd colomn - the class (what the model detected) e.g. a person, an aplle..

            //if a **person** was detected with high enough confidence (>0.5) then we will say a person was detected
            if (confidence > 0.5 && class_id == 15) { //class 15 is person, and we are interested in people
                human_detected_current = true; //Mark the flag
                //the detected person bounding box
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * W);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * H);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * W);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * H);

                //center of the person
                float cx = (x1 + x2) * 0.5f;
                float cy = (y1 + y2) * 0.55f; // ~the belly (we aim for the belly)
                if (smooth_x < 0) { //initialize smoothe_x and smooth_y (at first detection)
                    smooth_x = cx; 
                    smooth_y = cy;
                }
                else { //calculate the smmooth_x and smooth_y (not first detection)
                    smooth_x = ALPHA * smooth_x + (1 - ALPHA) * cx; 
                    smooth_y = ALPHA * smooth_y + (1 - ALPHA) * cy; 
                }
                target = cv::Point((int)smooth_x, (int)smooth_y); //the targer (~belly of detected person)
                //now we actually defining the bounding box
                cv::rectangle(disp, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                break; //after one person was detected we whould break (won't want to give the drones commands for moving to 2 different places)
            }
        }

        // Handle recording based on human detection
        if (human_detected_current) {
            frames_without_human = 0; //reset the var
            if (!recorder.isRecording()) { //if not recording ---> start record
                recorder.startRecording(W, H);
            }
        }
        else {
            frames_without_human++; //update the counter
            //if enough time passed since last detection (>3sec) ---> stop recording
            if (recorder.isRecording() && frames_without_human > STOP_RECORDING_DELAY) {
                recorder.stopRecording();
            }
        }

        // Add recording status to display
        if (recorder.isRecording()) {
            //text saying recording + a circle
            //some people says it gives the video the "Wow effect"
            //I hope you will think that too
            cv::Scalar rec_color = cv::Scalar(0, 255, 0) //choosing a color (we changed it, now only greenish (lime))
            cv::circle(disp, cv::Point(20, 20), 8, rec_color, -1); //creating a circle
            cv::putText(disp, "REC", cv::Point(35, 25), cv::FONT_HERSHEY_SIMPLEX, 0.5, rec_color, 2); //add the text
        }

        if (target.x >= 0) { //if a person was detected
            // Calculate and send angle
            float dx = smooth_x - W * 0.5f; //how far (left or right) the detected person is from the center
            float angle = atan2(dx, fx_cam) * 180.0f / CV_PI; //calculating the angle (of detected person), and converting radians to degs
            char angle_buffer[64]; //buffer for the angle (64 chars)
            std::snprintf(angle_buffer, sizeof(angle_buffer), "%.6f\n", (double)angle); //casting the angle to double, taking 6 digs after decimil point
            angleSend.sendMessage(string(angle_buffer)); //sending the angle in the UDP socket! finnalyyyy!

            //displaying the angle
            cv::circle(disp, target, 6, cv::Scalar(0, 255, 255), -1);
            string text = "Yaw: " + std::to_string(angle) + " deg";
            cv::putText(disp, text, cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(255, 0, 0), 2);
        }

        //add frame to recording if recording is active
        if (recorder.isRecording()) {
            recorder.addFrame(disp);
        }

        //displaying the image
        cv::imshow("MonoLive", disp);

        // Handle keyboard input
        int key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC key
            break;
        }

        human_detected_prev = human_detected_current; //update
    }

    // Stop recording before shutdown
    if (recorder.isRecording()) {
        recorder.stopRecording();
    }

    cout << "Shutting down..." << endl;
    //// SLAM.Shutdown();
    //// SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");
    return 0;
}