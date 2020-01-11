import tensorflow as tf

'''
Option 1 - dynamically build graph or something
'''
class Node:  # a node in the tree
  def __init__(self, ...):
    self.isLeaf = True / False
    self.hidden_state = None
    # for leaves
    self.word = word
    # for inner nodes
    self.left = None  # reference to left child
    self.right = None  # reference to right child
    

class RNN_Model():
    def add_model_vars(self):
        with tf.variable_scope('Embeddings'):
          embeddings = \
            tf.get_variable('embeddings', [len(self.vocab), self.config.embed_size])
        with tf.variable_scope('Composition'):
          W1 = tf.get_variable('W1',
                          [2 * self.config.embed_size, self.config.embed_size])
          b1 = tf.get_variable('b1', [1, self.config.embed_size])
        with tf.variable_scope('Projection'):
          U = tf.get_variable('U', [self.config.embed_size, self.config.label_size])
          bs = tf.get_variable('bs', [1, self.config.label_size])
      
    def build_subtree_model(node):
        if node.isLeaf:
          # lookup word embeddings
          node.hidden_state = tf.nn.embedding_lookup(embeddings,
                                                     vocab.encode(node.word))
        else:
          # build the model recursively and combine children nodes
          left_tensor = build_subtree_model(node.left)
          right_tensor = build_subtree_model(node.right)
          node.hidden_state = tf.nn.relu(tf.matmul(tf.concat(1, [left_tensor, right_tensor]), W1) + b1)
        return node.hidden_state
        
    def run_training_or_inference(self, input_trees):
        for i in xrange(INPUT_BATCHES):
          with tf.Graph().as_default(), tf.Session() as sess:
            self.add_model_vars()
            saver = tf.train.Saver()
            saver.restore(sess, backup_path)
            for tree in trees[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]:
                ### run training or inference ###
                saver.save(sess, backup_path) # in case of training, save progress
    
...
            
'''
Option 2 - using tf batch or sometihng
'''
embeddings, W1, b1, U, bs = # dunno what to puth ere....

vocab = {'the': 0, 'old': 1, 'cat': 2}
node_words = ['the', 'old', 'cat', '', '']
is_leaf = [True, True, True, False, False]
left_children = [-1, -1, -1, 1, 0]   # indices of left children nodes in this list
right_children = [-1, -1, -1, 2, 3]  # indices of right children nodes in this list

node_word_indices = [vocab[word] if word else -1 for word in node_words]

node_tensors = tf.TensorArray(tf.float32, size=0, dynamic_size=True,
                              clear_after_read=False, infer_shape=False)

def embed_word(word_index):
  with tf.device('/cpu:0'):
    return tf.expand_dims(tf.gather(embeddings, word_index), 0)

def combine_children(left_tensor, right_tensor):
  return tf.nn.relu(tf.matmul(tf.concat(1, [left_tensor, right_tensor]), W1) + b1)

def loop_body(node_tensors, i):
  node_is_leaf = tf.gather(is_leaf, i)
  node_word_index = tf.gather(node_word_indices, i)
  left_child = tf.gather(left_children, i)
  right_child = tf.gather(right_children, i)
  node_tensor = tf.cond(
      node_is_leaf,
      lambda: embed_word(node_word_index),
      lambda: combine_children(node_tensors.read(left_child),
                               node_tensors.read(right_child)))
  node_tensors = node_tensors.write(i, node_tensor)
  i = tf.add(i, 1)
  return node_tensors, i

loop_cond = lambda node_tensors, i: \
        tf.less(i, tf.squeeze(tf.shape(is_leaf)))

node_tensors, _ = tf.while_loop(loop_cond, loop_body, [node_tensors, 0],
                                     parallel_iterations=1)
                                     
                                     
                                     
                                     
                                     