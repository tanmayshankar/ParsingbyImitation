
		############## THIS IS HOW TO USE THE CASE: 
		y = tf.placeholder(tf.float32,shape=[3])

		# Based on the value of this index, we choose which thing to use. 
		index = tf.placeholder(tf.int32)		#IMPORTANT: MAKE SURE YOU DON'T ASSIGN SHAPE TO IT. 
		#The tf.case( pred) requires pred to have no shape (rank 0 tensor).

		def case1():
			w1 = tf.ones(3)
			return tf.multiply(w1,y)

		def case2():
			w2 = 2*tf.ones(3)
			return tf.multiply(w2,y)

		def case3():
			w3 = 3*tf.ones(3)
			return tf.multiply(w3,y)

		def case4():
			# Default
			return -tf.ones(3)

		output = tf.case({tf.equal(index,1):case1, tf.equal(index,2):case2, tf.equal(index,3):case3}, default=case4, exclusive=True)

