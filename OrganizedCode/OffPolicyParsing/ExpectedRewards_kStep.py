
# coding: utf-8

# In[1]:


import numpy as npy, os, matplotlib.pyplot as plt


# In[2]:


cd ../../ImageSets/


# In[3]:


images = npy.load("NEWSTRIPS.npy")
images[images==2]=1.
images[images==-1]=0.

labels = npy.load("NEWSTRIPS.npy")


# In[4]:


def infogain(y,a,l):
#     if a==1:
#         s1=y[:l,:]
#         s2=y[l:,:]        
#     if a==0:
#         s1=y[:,:l]
#         s2=y[:,l:]
    if a==0:
        s1=y[:l,:]
        s2=y[l:,:]        
    if a==1:
        s1=y[:,:l]
        s2=y[:,l:]

    ysize = y.shape[0]*y.shape[1]
    ps1 = float( s1.shape[0]*s1.shape[1])/ysize
    ps2 = float(s2.shape[0]*s2.shape[1])/ysize
    return entropy(y)-(ps1*entropy(s1)+ps2*entropy(s2))


# In[5]:


def entropy(y):     
    nones = npy.count_nonzero(y)
    ysize = y.shape[0]*y.shape[1]    
    nzeros = ysize-nones
    pz = float(nzeros)/ysize
    po = float(nones)/ysize
    if pz==0 or po==0: 
        return 0.                
    return -(pz*npy.log2(pz)+po*npy.log2(po))  


# In[6]:


def bestsplit(x):
    maxval = -1           
    chosen_a = -1                             
    chosen_l = -1                                       
    print("Shape of image:",x.shape)
    for a_val in range(2):         
        if a_val==0:
            limval = x.shape[0]-1
        if a_val==1:
            limval = x.shape[1]-1
        for l_val in range(1,limval):
            
            ig = infogain(x,a_val,l_val)   
#             print(a_val,l_val,ig,limval, chosen_a,chosen_l, maxval)
            if ig>maxval:            
                maxval=ig                          
                chosen_a = a_val                  
                chosen_l = l_val   
    if maxval==0:
        print("No entropy reducing splits.")
        return 1
    return chosen_a,chosen_l


# In[7]:


def return_splits(x,a,l):
    if a==0:
        s1=x[:l,:]
        s2=x[l:,:]
    if a==1:
        s1=x[:,:l]
        s2=x[:,l:]
    return s1,s2


# In[33]:


r_values = npy.zeros(500)

# For all images:
for i in range(500):
    a,l = bestsplit(images[i])
    s1,s2 = return_splits(labels[i],a,l)
    
    s1_sum = s1.sum()
    if s1_sum==0:
        s1_val = 0
    else:
        s1_val = (s1_sum/abs(s1_sum))*s1_sum
    
    s2_sum = s2.sum()
    if s2_sum==0:
        s2_val = 0
    else:
        s2_val = (s2_sum/abs(s2_sum))*s2_sum          
    
    r_values[i] = (s1_val+s2_val)/(256**2)
    
    print(i)


# In[39]:


from sklearn.metrics import jaccard_similarity_score
j_values = npy.zeros(500)

# For all images:
for i in range(500):
    a,l = bestsplit(images[i])
    s1,s2 = return_splits(labels[i],a,l)
    preds = npy.zeros((256,256))
    s1_mean = s1.mean()
    if s1_mean==0:
        preds[:l,:] = 1.
    else:
        preds[:l,:] = s1_mean/abs(s1_mean)
    s2_mean = s2.mean()
    if s2_mean==0:
        preds[l:,:] = 1.
    else:
        preds[l:,:] = s2_mean/abs(s2_mean)
        
    j_values[i] = jaccard_similarity_score(labels[i].reshape((256**2)),preds.reshape((256**2)))
    print(i)


# In[8]:


# Under a random split location policy, with optimal assignments.


random_r_values = npy.zeros((500,255))

# For all images:
for i in range(500):
    
    for l in range(255):
        
        s1, s2 = return_splits(labels[i],0,l)
    
        s1_sum = s1.sum()
        if s1_sum==0:
            s1_val = 0
        else:
            s1_val = (s1_sum/abs(s1_sum))*s1_sum

        s2_sum = s2.sum()
        if s2_sum==0:
            s2_val = 0
        else:
            s2_val = (s2_sum/abs(s2_sum))*s2_sum          
    
        random_r_values[i,l] = (s1_val+s2_val)/(256**2)
    
    print(i)


# In[10]:


random_r_values.mean()


# In[40]:


j_values


# In[41]:


j_values.mean()


# In[35]:


r_values.mean()


# In[38]:


tanvalues = npy.zeros((500))

for i in range(500):
    tanvalues[i] = npy.math.tan(r_values[i])
tanvalues,tanvalues.mean()


# In[ ]:


a,l = bestsplit(s2)
s1,s2 = return_splits(s2,a,l)


# In[ ]:


plt.imshow(s1)
plt.show()


# In[ ]:


a,l = bestsplit(s1)
s1,s2 = return_splits(s1,a,l)


# In[ ]:


plt.imshow(s2)
plt.show()


# In[ ]:


a,l = bestsplit(s2)
s1,s2 = return_splits(s2,a,l)


# In[ ]:


plt.imshow(s1)
plt.show()


# In[ ]:


a,l = bestsplit(s1)
s3,s4 = return_splits(s1,a,l)


# In[ ]:


plt.imshow(s3)
plt.show()


# In[ ]:


plt.imshow(s4)
plt.show()


# In[ ]:


a,l = bestsplit(s3)
a,l

