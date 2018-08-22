# Learning Neural Parsers with Deterministic Differentiable Imitation Learning

### What is this all about? ###
This repository is code connected to the paper - T. Shankar, N. Rhinehart, K. Muelling, K. M. Kitani, Learning Neural Parsers with Deterministic Differentiable Imitation Learning, submitted to the Conference on Robot Learning (CoRL) 2018.

### Where can I read this cool paper? ###
[Click here to view the paper on ArXiv!](https://arxiv.org/abs/1806.07822) 

### I want a TL;DR of what this repository does. ###
Our paper targets learning a parser to construct hierarchical decompositions of object images, by imitating a decision tree like algorithm. This repository implements the code for the following: 

1. Introduces a framework to learn to decompose spatial tasks into segments by parsing, motivated by the problem of a painting robot covering a large object.
2. Formulates object decomposition as a parsing problem, inspired by the similarity between parse-trees of objects and structured decisions trees constructed by ID3.
3. Trains a neural object parser to construct hierarchical object decompositions by imitating a ground-truth expert, in the form of a information gain maximizing ID3 like algorithm.
4. Introduces a novel hybrid imitation-reinforcement learning approach, by building a deterministic DDPG-style actor-critic variant of AggreVaTeD.

### Is that all? ###
Yes and no. I'm evaluating our novel policy gradient update, DRAG, on OpenAI Gym environments! you can check out https://github.com/tanmayshankar/DeepVectorPolicyFields for more! 

### Can I use this code to pretend I did some research? ###
Feel free to use my code, but please remember to cite my paper above (and this repository)! 

### I have a brilliant idea to make this even cooler! ###
Awesome! Feel free to mail me at tanmay.shankar@gmail.com with your suggestions, feedback and ideas! 





