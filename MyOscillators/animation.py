import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import sys

def load_trajectory(filepath):
    """ Load the trajectory data from an .npy file. """
    return np.load(filepath)

def create_animation(trajectory, filepath_output):
    """ Create and save an animation of the oscillator trajectories. """
    fig, ax = plt.subplots()
    num_frames, num_oscillators = trajectory.shape
    ax.set_xlim(-1, num_oscillators)
    ax.set_ylim(np.min(trajectory) - 1, np.max(trajectory) + 1)
    
    points = [ax.plot([], [], 'o')[0] for _ in range(num_oscillators)]
    
    def init():
        """ Initialize the animation with empty data. """
        for point in points:
            point.set_data([], [])
        return points
    
    def update(frame):
        """ Update the positions of the oscillators for each frame. """
        for i, point in enumerate(points):
            point.set_data(i, trajectory[frame, i])
        return points
    
    # Reduce frame rate if necessary by skipping frames
    frame_skip = max(1, num_frames // 100)  # Adjust based on performance needs
    ani = animation.FuncAnimation(fig, update, frames=range(0, num_frames, frame_skip),
                                  init_func=init, blit=True, interval=50)
    
    # Save the animation as a GIF
    ani.save(filepath_output, writer=FFMpegWriter(fps=15))

# Example usage
if __name__ == "__main__":
    trajectory_path = str(sys.argv[1])  # Set the path to your .npy file
    output_path = 'oscillator_animation.mp4'  # Set the output file path
    trajectory_data = load_trajectory(trajectory_path)
    create_animation(trajectory_data, output_path)
