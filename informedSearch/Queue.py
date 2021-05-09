"""
Alex Staley -- 919519311
CS441 -- Program 1
January 31, 2021

This file implements the Queue class used to maintain the
priority queue for determining which node to expand first.

The queue is ordered from lowest- to highest-priority, according
to the heuristic passed to the applicable insert function.
"""

MAXQUEUE = 10


class Queue:
    def __init__(self):
        self.queue = []

    def pop(self, index):
        return self.queue.pop(index)

    def isEmpty(self):
        return self.queue == []

    def length(self):
        return len(self.queue)

    def getBest(self, explored):
        """
        Get the node on the queue with the highest priority.
        The queue is ordered from lowest priority to highest.
        :param explored: List of explored nodes to avoid
        :return: Highest priority node that hasn't been explored
        """
        # Search for first (i.e. last) node in the queue
        for option in range(self.length() - 1, -1, -1):
            found = False
            for expNode in explored:

                # If a node in the queue is found in the explored list,
                # Remove it from the queue and check the next node
                if expNode.isEqual(self.queue[option]):
                    found = True
                    self.pop(option)
                    break

            # Use the first node found outside the explored list
            if not found:
                break
        return self.pop(option)

    def insertBestFirst(self, newNode, heuristic):
        """
        Insert a node into the priority queue, following the
        Best-first algorithm and using the given heuristic
        :param newNode: The node to be inserted
        :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
        :return: No return value
        """
        # Insert to an empty queue
        if self.isEmpty():
            self.queue.insert(0, newNode)
            return

        # Use the Best-first algorithm to order the queue,
        # along with the given heuristic
        index = 0
        for item in self.queue:
            if item.isOverBFS(newNode, heuristic):
                index += 1
            else:
                break
        self.queue.insert(index, newNode)

        # Keep queue at a maximum length of MAXQUEUE
        if self.length() > MAXQUEUE:
            self.pop(0)

    def insertAStar(self, newNode, heuristic):
        """
        Insert a node into the priority queue, following the
        A* algorithm and using the given heuristic
        :param newNode: The node to be inserted
        :param heuristic: 1 == Manhattan Distance; 2 == Misplaced Tiles; 3 == Inversions
        :return: No return value
        """
        # Insert to an empty queue
        if self.isEmpty():
            self.queue.insert(0, newNode)
            return

        # Use the A* algorithm to order the queue,
        # along with the given heuristic
        index = 0
        for item in self.queue:
            if item.isOverAS(newNode, heuristic):
                index += 1
            else:
                break
        self.queue.insert(index, newNode)

        # Keep queue at a maximum length of MAXQUEUE
        if self.length() > MAXQUEUE:
            self.pop(0)
