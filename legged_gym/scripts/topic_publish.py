import rospy
from std_msgs.msg import Float32MultiArray
from multiprocessing import shared_memory
import numpy as np

def publisher():
    # Initialize ROS node
    rospy.init_node('shm_data_publisher', anonymous=True)
    # Create a publisher on a given topic
    pub = rospy.Publisher('shared_memory_topic', Float32MultiArray, queue_size=10)
    rate = rospy.Rate(10)  # 10 Hz

    # Access the shared memory block
    shm_name = 'observation'
    shm = shared_memory.SharedMemory(name=shm_name)
    x = np.zeros(48, dtype=np.float64)
    data_bytes = x.tobytes()

    try:
        while not rospy.is_shutdown():
            # Read data from shared memory
            data_bytes = shm.buf[:len(data_bytes)]  # Adjust slice as needed
            data = np.frombuffer(data_bytes, dtype=np.float64)  # Adjust dtype as per your data

            # Create a message and publish it
            msg = Float32MultiArray(data=data)
            pub.publish(msg)

            rate.sleep()
    finally:
        # Clean up shared memory access
        shm.close()

if __name__ == '__main__':
    try:
        publisher()
    except rospy.ROSInterruptException:
        pass
