#!/usr/bin/env python
from headers import *

def parse_arguments():
	parser = argparse.ArgumentParser(description='Primitive-Aware Segmentation Argument Parsing')
	parser.add_argument('--images',dest='images',type=str)
	parser.add_argument('--labels',dest='labels',type=str)
	parser.add_argument('--paintwidth',dest='paintwidth',type=int)
	return parser.parse_args()

def main(args):

	args = parse_arguments()
	images = npy.load(args.images)
	labels = npy.load(args.labels)
	paintwidth = args.paintwidth
	number_images = 163

	downscaled_labels = copy.deepcopy(labels)
	downscaled_labels = npy.zeros((number_images,int(labels[0].shape[0]/paintwidth),int(labels[0].shape[1]/paintwidth)))
	new_images = npy.zeros((number_images,int(labels[0].shape[0]/paintwidth),int(labels[0].shape[1]/paintwidth),3))
	labels[labels==-1] = 0.

	# Planning for all images.
	# for i in range(images.shape[0]):
	for i in range(1):
	# for i in range(230,images.shape[0]):
		print "___________________________________________________________________________"		
		print "______________________________IMAGE {0}____________________________________".format(i)
		# First downsample.
		print "Downsampling."
		
		# downscaled_labels[i] = cv2.resize(downscaled_labels[i], (int(labels[i].shape[0]/paintwidth), int(labels[i].shape[1]/paintwidth)), interpolation=cv2.INTER_NEAREST)
		downscaled_labels[i] = cv2.resize(labels[i], (int(labels[i].shape[0]/paintwidth), int(labels[i].shape[1]/paintwidth)), interpolation=cv2.INTER_NEAREST)
		new_images[i] = cv2.resize(images[i], (int(labels[i].shape[0]/paintwidth), int(labels[i].shape[1]/paintwidth)))

		num_nodes = npy.count_nonzero(downscaled_labels[i])
		coords = npy.zeros((num_nodes,2))
		indices = npy.where(downscaled_labels[i]==1.)
		# print(indices)
		for j in range(num_nodes):
			coords[j] = [indices[0][j],indices[1][j]]

		print "Constructing KD Tree."
		kdtree = spatial.KDTree(coords)

		distances = 10000*npy.ones((num_nodes,num_nodes))
		adjacency = npy.zeros((num_nodes,num_nodes))

		print "Converting to Grid Roadmap."
		for j in range(num_nodes):
			dists, inds = kdtree.query(coords[j],p=1,distance_upper_bound=1.1,k=5)
			for k in range(len(inds)):
				if inds[k]<num_nodes:
					# adjacency[i,inds[k]] = 1.			
					distances[j,inds[k]] = dists[k]
		for j in range(num_nodes):
			distances[j,j] = 10000

		# embed()
		print "Solving TSP."
		path = solve_tsp(distances=distan ces)
		# path = solve_tsp(distances=distances,endpoints=(0,num_nodes-1))

		print "Plotting."
		fig, ax = plt.subplots(1,2,sharey=True)
		fig.show()

		ax[0].imshow(new_images[i],origin='lower')
		ax[0].set_adjustable('box-forced')
		ax[1].imshow(downscaled_labels[i],origin='lower')
		ax[1].set_adjustable('box-forced')
		ax[0].scatter(coords[:,1],coords[:,0],edgecolors='k',c=npy.argsort(path))
		ax[1].scatter(coords[:,1],coords[:,0],edgecolors='k',c=npy.argsort(path))		
		
		# # print(path)
		for j in range(len(path)-1):
			ax[0].plot([coords[path[j],1],coords[path[j+1],1]],[coords[path[j],0],coords[path[j+1],0]],'b')
			ax[1].plot([coords[path[j],1],coords[path[j+1],1]],[coords[path[j],0],coords[path[j+1],0]],'b')

		plt.show()

if __name__ == '__main__':
	main(sys.argv)