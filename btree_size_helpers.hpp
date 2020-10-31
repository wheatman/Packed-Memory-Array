#pragma once

#include "tlx/container/btree.hpp"

// copied from the tlx source code so we can get access to the size of the nodes
// to meausre the total memory usage

template <typename Key, typename Value, typename KeyOfValue,
          typename Compare = std::less<Key>,
          typename Traits = tlx::btree_default_traits<Key, Value>,
          bool Duplicates = false, typename Allocator = std::allocator<Value>>
class BTree_size_helper {
public:
  //! \name Template Parameter Types
  //! \{

  //! First template parameter: The key type of the B+ tree. This is stored in
  //! inner nodes.
  typedef Key key_type;

  //! Second template parameter: Composition pair of key and data types, or
  //! just the key for set containers. This data type is stored in the leaves.
  typedef Value value_type;

  //! Third template: key extractor class to pull key_type from value_type.
  typedef KeyOfValue key_of_value;

  //! Fourth template parameter: key_type comparison function object
  typedef Compare key_compare;

  //! Fifth template parameter: Traits object used to define more parameters
  //! of the B+ tree
  typedef Traits traits;

  //! Sixth template parameter: Allow duplicate keys in the B+ tree. Used to
  //! implement multiset and multimap.
  static const bool allow_duplicates = Duplicates;

  //! Seventh template parameter: STL allocator for tree nodes
  typedef Allocator allocator_type;

  //! \}

public:
  //! \name Constructed Types
  //! \{

  //! Typedef of our own type
  typedef BTree_size_helper<key_type, value_type, key_of_value, key_compare,
                            traits, allow_duplicates, allocator_type>
      Self;

  //! Size type used to count keys
  typedef size_t size_type;

  //! \}

public:
  //! \name Static Constant Options and Values of the B+ Tree
  //! \{

  //! Base B+ tree parameter: The number of key/data slots in each leaf
  static const unsigned short leaf_slotmax = traits::leaf_slots;

  //! Base B+ tree parameter: The number of key slots in each inner node,
  //! this can differ from slots in each leaf.
  static const unsigned short inner_slotmax = traits::inner_slots;

  //! Computed B+ tree parameter: The minimum number of key/data slots used
  //! in a leaf. If fewer slots are used, the leaf will be merged or slots
  //! shifted from it's siblings.
  static const unsigned short leaf_slotmin = (leaf_slotmax / 2);

  //! Computed B+ tree parameter: The minimum number of key slots used
  //! in an inner node. If fewer slots are used, the inner node will be
  //! merged or slots shifted from it's siblings.
  static const unsigned short inner_slotmin = (inner_slotmax / 2);

  //! Debug parameter: Enables expensive and thorough checking of the B+ tree
  //! invariants after each insert/erase operation.
  static const bool self_verify = traits::self_verify;

  //! Debug parameter: Prints out lots of debug information about how the
  //! algorithms change the tree. Requires the header file to be compiled
  //! with TLX_BTREE_DEBUG and the key type must be std::ostream printable.
  static const bool debug = traits::debug;

  //! \}

public:
  //! \name Node Classes for In-Memory Nodes
  //! \{

  //! The header structure of each node in-memory. This structure is extended
  //! by InnerNode or LeafNode.
  struct node {
    //! Level in the b-tree, if level == 0 -> leaf node
    unsigned short level;

    //! Number of key slotuse use, so the number of valid children or data
    //! pointers
    unsigned short slotuse;

    //! Delayed initialisation of constructed node.
    void initialize(const unsigned short l) {
      level = l;
      slotuse = 0;
    }

    //! True if this is a leaf node.
    bool is_leafnode() const { return (level == 0); }
  };

  //! Extended structure of a inner node in-memory. Contains only keys and no
  //! data items.
  struct InnerNode : public node {
    //! Define an related allocator for the InnerNode structs.
    typedef typename std::allocator_traits<Allocator>::template rebind_alloc<
        InnerNode>
        alloc_type;

    //! Keys of children or data pointers
    key_type slotkey[inner_slotmax]; // NOLINT

    //! Pointers to children
    node *childid[inner_slotmax + 1]; // NOLINT

    //! Set variables to initial values.
    void initialize(const unsigned short l) { node::initialize(l); }

    //! Return key in slot s
    const key_type &key(size_t s) const { return slotkey[s]; }

    //! True if the node's slots are full.
    bool is_full() const { return (node::slotuse == inner_slotmax); }

    //! True if few used entries, less than half full.
    bool is_few() const { return (node::slotuse <= inner_slotmin); }

    //! True if node has too few entries.
    bool is_underflow() const { return (node::slotuse < inner_slotmin); }
  };

  //! Extended structure of a leaf node in memory. Contains pairs of keys and
  //! data items. Key and data slots are kept together in value_type.
  struct LeafNode : public node {
    //! Define an related allocator for the LeafNode structs.
    typedef typename std::allocator_traits<Allocator>::template rebind_alloc<
        LeafNode>
        alloc_type;

    //! Double linked list pointers to traverse the leaves
    LeafNode *prev_leaf;

    //! Double linked list pointers to traverse the leaves
    LeafNode *next_leaf;

    //! Array of (key, data) pairs
    value_type slotdata[leaf_slotmax]; // NOLINT

    //! Set variables to initial values
    void initialize() {
      node::initialize(0);
      prev_leaf = next_leaf = nullptr;
    }

    //! Return key in slot s.
    const key_type &key(size_t s) const {
      return key_of_value::get(slotdata[s]);
    }

    //! True if the node's slots are full.
    bool is_full() const { return (node::slotuse == leaf_slotmax); }

    //! True if few used entries, less than half full.
    bool is_few() const { return (node::slotuse <= leaf_slotmin); }

    //! True if node has too few entries.
    bool is_underflow() const { return (node::slotuse < leaf_slotmin); }

    //! Set the (key,data) pair in slot. Overloaded function used by
    //! bulk_load().
    void set_slot(unsigned short slot, const value_type &value) {
      TLX_BTREE_ASSERT(slot < node::slotuse);
      slotdata[slot] = value;
    }
  };
};
