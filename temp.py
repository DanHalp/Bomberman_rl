from sklearn . datasets import load_digits
import matplotlib.pyplot as plt
import numpy as np

#  = load_digits ()
# # data = digits ["data"]
# # i# digitsmages = digits ["images"]
# target = digits ["target"]
# target_names = digits ["target_names"]

# def picShift(pic, reduce=0):
#     edge = len(pic) - 1 -  reduce
#     if edge < 4:
#         return
#     dif = edge - reduce
#
#     for i in range(dif):
#         temp = [pic[reduce + i, reduce], pic[reduce + dif, reduce + i], pic[reduce + dif - i, reduce + dif], pic[reduce, reduce + dif - i]]
#         pic[reduce + i, reduce] = temp[3]
#         pic[reduce + dif, reduce + i] = temp[0]
#         pic[reduce + dif - i, reduce + dif] = temp[1]
#         pic[reduce, reduce + dif - i] = temp[2]
#
#     picShift(pic, reduce + 1)
#
#
# x = np.array(images[7])
# # plt.figure ()
# # plt.gray ()
# # plt.subplot(122)
# # plt.imshow (x , interpolation ="nearest") # also try interpolation =" bicubic "
# # plt.show()
#
#
# picShift(x)
# plt.figure ()
# plt.gray ()
# plt.subplot(122)
# plt.imshow (x , interpolation ="nearest") # also try interpolation =" bicubic "
# plt.show ()


class BinarySearchTree:
    def __init__(self, value):
        self.head = Node(value)

    def add(self, value):
        curr = self.head
        while True:
            if value < curr.data:
                if curr.left is None:
                    curr.left = Node(value)
                    break
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = Node(value)
                    break
                curr = curr.right

    def find_lowest_parent(self, n1, n2):

        if self.head is None:
            return

        if self.head.data == n1 or self.head.data == n2:
            return self.head.data

        parent, d = [-1], [np.inf]
        self.util(self.head, n1, n2, parent, [], d)
        return parent[0]

    def util(self, curr,n1,n2,parent,counter, depth, d=0):

        if curr is None or len(counter) == 2:
            return

        if curr.data == n1 or curr.data == n2:
            counter.append(curr.data)
            if len(counter) < 2:
                parent[0] = curr.data
                depth[0] = d

        self.util(curr.left, n1, n2, parent,counter,  depth, d+1)
        if d < depth[0] and len(counter) < 2:
            parent[0] = curr.data
            depth[0] = d
        self.util(curr.right, n1, n2, parent, counter, depth, d+1)


    def findSum(self, sum):

        if self.head is None:
            return "Tree is empty...DO something about it."
        self.findSumUtil(self.head, sum)

    def findSumUtil(self, curr, sum, buffer=list(), uptonow=0):

        buffer = buffer + [curr.data]
        uptonow += curr.data
        if uptonow >= sum:
            curr_sum = 0
            for i in range(len(buffer) - 1, -1, -1):
                curr_sum += buffer[i]
                if curr_sum == sum:
                    print(buffer[i:])
                    break
                elif curr_sum > sum:
                    break

        if curr.left is not None:
            self.findSumUtil(curr.left, sum, buffer, uptonow)

        if curr.right is not None:
            self.findSumUtil(curr.right, sum, buffer, uptonow)




class Node:

    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None


def main():
    # tree = BinarySearchTree(5)
    # tree.add(2)
    # tree.add(8)
    # tree.add(1)
    # tree.add(3)
    # tree.add(6)
    # tree.add(9)
    # tree.add(4)
    # tree.add(7)
    # tree.add(10)
    #
    # tree.findSum(14)

    x = [500, 1500,3000, 12000, 21500]
    y = [np.mean([160, 34, 185, 15, 168]), np.mean([20, 218, 151, 26, 147]),
        np.mean([149, 400, 15, 114, 73]), np.mean([400, 37, 400, 400, 50,400]),
         np.mean([400, 400, 400, 400, 127,400])]
    plt.subplots(figsize=(8,8))
    plt.plot(x, y, "orange")
    plt.scatter(x,y, s=50, c="red", alpha=0.7)

    plt.title("Average Steps  / Training Games")
    plt.xlabel("Number of training games")
    plt.ylabel("Average Steps")
    plt.show()
main()