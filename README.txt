################# PROGRAM STRUCTURE #####################
main.py is the executable file for running the program.
It takes command line arguments and creates an instance of manufmodel.
manufmodel is the object that basically is the simulation.
It contains all of the information used to track system state.
It is also the superclass for robot, blade, and station - a manufmodel contains all of these objects representing the whole system.

Due to a poor choice of packages to use, a lot of the code in these files is just aimed towards making the simulation and visualization run smoothly.
However, robot.py contains a lot of important code related to the reinforcement learning algorithm.
All calculations directly related to the RL algorithm are in the decision.py file.
Each of these files contains more detailed information in the headers and inline comments.

Visualization.py is a standalone file used for running the simulation visualization in a web browser.
It has no bearing on the algorithm or simulation and the code inside can be disregarded.

################# TRAINING THE ALGORITHM #################
I have yet to implement a comprehensive argparser so for the time being, to train the algorithm run the command
'python manufmodel.py blades values.npy' where blades is the number of blades you want to train on and the optional values.npy is the numpy file that contains a state-action pair array.
I included this in the event that you train the algorithm and it doesn't produce the intended effects, you can continue training where you left off instead of starting over because it writes Q_sa to a file.
If you want to try batching instead of random arrivals you can open the manufmodel file and uncomment the lines in the generate_arrival_times() function.
Similarly, if you want to change the number of episodes for training you can change the bounds on the loop in the main function (this will all be functionality that a unified argparse will take care of in the near future)

######### RUNNING THE SIMULATION VISUALIZATION ############
To run the simulation visualization you need to specify a policy to follow.
The command to do so will be 'python visualization.py policy.out' where policy.out is the policy.
A policy file is generated after training is completed.
