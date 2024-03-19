import rospy
from std_msgs.msg import Float32MultiArray
from multiprocessing import shared_memory
from svan_msgs.msg import JointState
import numpy as np

joint_pos = np.zeros(12)

curr = 0

def get_position_data(msg):
    global joint_pos
    global curr
    joint_pos = np.array(msg.position, dtype = np.float64)
    # print(curr)
    curr+=1
    # print("Incoming Data: ", joint_pos, "curr ", curr)

def publisher():
    # Initialize ROS node
    rospy.init_node('shm_data_publisher', anonymous=True)
    # Create a publisher on a given topic
    pub = rospy.Publisher('/motor_command_reflect', Float32MultiArray, queue_size=10)
    sub = rospy.Subscriber('/svan/motor_command', JointState, get_position_data)
    rate = rospy.Rate(100)  # 10 Hz

    # Access the shared memory block
    shm_name = 'joint_state'
    shm_size = 1024
    shm = shared_memory.SharedMemory(create=True, name=shm_name, size=shm_size)
    x = np.zeros(12, dtype=np.float64)
    data_bytes = x.tobytes()
    current_mean = 0 
    prev_val = []
    try:
        while not rospy.is_shutdown():
            # Read data from shared memory
            data_bytes = joint_pos.tobytes()
            shm.buf[:len(data_bytes)] = data_bytes # Adjust slice as needed
            
            data_bytes_2 = shm.buf[:len(data_bytes)]  # Adjust slice as needed
            data = np.frombuffer(data_bytes_2, dtype=np.float64)  # Adjust dtype as per your data
            current_mean = np.mean(data)
            prev_val = data
            # print("Data Value: ", data)
            
            # Create a message and publish it
            # msg = Float32MultiArray(data=data)
            # pub.publish(msg)

            rate.sleep()
    finally:
        # Clean up shared memory access
        shm.close()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
