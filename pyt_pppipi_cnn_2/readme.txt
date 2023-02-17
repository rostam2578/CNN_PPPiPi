DGLBhabhaGcnReNewestweight7N:
In this new version of 7, I would like to explore the
hyperspace more. So a better range of weight decay and learning
rate is explored using the tensorboard tools. I first I ran the 
model for batch size of 5. The best result for 5000 training events
and 10 epochs is for lr-decay= 0.001, 0.001 with eff-pur of 99%.
We run this for 10000 events and 200 epochs. We can later run it for 
other sets of events.
 Also a better use of 
checkpoint and model saving is provided.


DGLBhabhaGcnReNewestweight7:
We continue by adding message passing process to the result of
the 3rd conv layer. The result is close to before (96%) for
about 200 epochs. I am trying 500 epochs now.


DGLBhabhaGcnReNewestweight6:
The results of message paaing is not nice, so let's go back to
the original model. The only difference from the origin which
used to give us 90% is to have different node features for all
conv nodes at any layer and let them be trained.
result:
the presence of the relu is essensil in the model. By having 
message passing after only the first two conv layers and controling 
all the computations by relu and training with about 200 epochs, 
we reach to 95%. Further training even yealds 97%.


DGLBhabhaGcnReNewestweight5:
Investigating how to make edge weights learnable. 

Hear the goal is to not use edge weights defined in dgl, but only 
a torch matrix introduced outside the graph which is introduced 
to the optimizer as a learnable parameter. The the challange will 
be to do message passing with it.

before that I am investigating what happens if I change the names 
of node features in the messaage passing for each layer and print 
everything to learn how this actualy happens. Maybe the way I did 
does not incorporates the edge weights.