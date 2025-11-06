# Deep-Learning
Deep learning course assignments and projects.
I inserted in the readme only the projects I liked the most.

## Linear Regression - Assignment 1 Part 2

The goal of this exercise was to implement a Linear Regression model and train it using the Gradient Descent algorithm.
We used the mean squared error (MSE) to calculate the loss: 
<br><img width="398" height="57" alt="image" src="https://github.com/user-attachments/assets/b58a2124-e9f1-4cbe-9d93-80c77131326f" /> <br>
Since the idea behind gradient descent is find the extremum of the loss, or where the loss stops decreasing - we'll derive the loss function with respect to the weights: 
<br><img width="496" height="66" alt="image" src="https://github.com/user-attachments/assets/21ba90e4-8fbf-4d24-b769-a1622c8c3dfd" /> <br>
Now we need to finetune the model learning rate and the epoches. 
When the learning rate is too low we might take steps too small (or don't learn enough between each epoch) and it will take too long to reach the minimum loss point. 
If the learning rate is too high we might skip the minimum loss point again and again, it can be seen when the loss jumps all over the places (and doesn't decrease slowly like it should).
This is an example of each "line" the model predicts after each a epoch when the learning rate is too small:
<br><img width="543" height="409" alt="image" src="https://github.com/user-attachments/assets/bffeb670-e7e2-460b-a9e7-f78e66fbe833" /><br>
When finetuning the number of epoches.
We discovered that too many epoches might repeat the same loss because we already got the desired model, and also take more time.
If we use too little and we won't even finish reaching the minimum loss.
This is the model progress after finetuning the learning rate and epoches:
<br><img width="534" height="410" alt="image" src="https://github.com/user-attachments/assets/0a418dee-0f84-4562-b585-29060cba5703" /><br>

### Technologies used: <b><code>Python</code></b>, <b><code>PyTorch</code></b>, <b><code>numpy</code></b>, <b><code>matplotlib</code></b>, <b><code>sklearn</code></b>.

## Convolution Neural Networks (CNNs) - Assignment 2 Part 1

The goal of this exercise was to get familiar with Convolutional Neural Networks and understand how different CNN architectures behave when trained on image datasets (in this case CIFAR-10).
Just CNNs are designed to solve problems which need matrix of datas (for example in this case images).
This is a simple example of some CIFAR-10 data:
<br><img width="466" height="357" alt="image" src="https://github.com/user-attachments/assets/1e1d076a-1ebd-44da-afe2-35ff633b90ae" /><br>
We start off by implementing an early stopping algorithm which simplifies the finetuning of epoches and and many times saves alot of times.
It works like this - if we see no (big enough) improvment of the loss after a number of epoches in a row we can assume we reached the maximum potential for the current model and stop the learning process early.
We will use this model architecture and experiment with the values N,P and M:
<br> <b><code>\[(CONV -> ReLU)*P -> MaxPool\]*(N/P) -> (Linear -> ReLU)*M -> Linear</code></b> <br>
Here N is the total number of convolutional layers, P specifies how many convolutions to perform before each pooling layer and M specifies the number of hidden fully-connected layers before the final output layer.
we start of trying out different number of layers (L), or in other words different model depths.
This is the result from the first experiment:
<br><img width="643" height="437" alt="image" src="https://github.com/user-attachments/assets/282dac2b-815a-478a-a433-da870232520d" /><br>
As we can see the "deeper" models performed badly, the reasons are the following:
Deeper models try to solve more complex problems, and by that ignoring easy solutions.
In NN we call the the vanishing gradient, when adjusting the weights we work backwards and each weight depends on the change of the weights in the layer ahead of him.
The problem is the change usually get smaller each layer and with alot of layers the weights on the first layers get almost no change hense the name the "vanishing gradient".
There could be more problems for example the weight are initniallized randomly and could cause the changes to be too big or too small and also cause the vanishing gradient or in the case they are too big the exploding gradient - when change gets too big in the first layers, the deeper models have more layers hense more sensitive to this problem. There are solutions for these problems which is different loss function such as ReLU or initializing the weights (not randomly).

lastly we experimented with different filters per layer amounts, with combinations of number of layers. 
This an example of the difference in filters per layers amount with 4 layers:
<br><img width="646" height="438" alt="image" src="https://github.com/user-attachments/assets/9f634c14-2dc9-4f6c-8fac-4afe17f8a22e" /><br>
1. More filters means more representational power
- Each filter captures a unique pattern.
- More filters means more information the model can extract at once.
- hense the model started learning faster — it could immediately detect a wide variety of patterns.
2. But also more filters means more parameters
- Every extra filter adds weights.
- Hense more computation per batch (slower training overall).
- And also higher risk of overfitting, which is why we can see the models with more filters finished learning earlier then the others.
We ended up with around 70% accuracy which 4 layers per convolution and 256 filters per layer.

### Technologies used: <b><code>Python</code></b>, <b><code>PyTorch</code></b>, <b><code>numpy</code></b>, <b><code>matplotlib</code></b>, <b><code>torch</code></b>, <b><code>torchvision</code></b>,<b><code>tqdm</code></b>.
