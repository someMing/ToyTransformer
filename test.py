# import numpy as np
#
# test_str_zh = "你好吗"
# test_str_en = "how are you?"
#
# print( list( test_str_zh ) )
# print( test_str_en.split() )
#
# test_list1 = [(0, '我'), (1, '爱'), (2, '你')]
# test_list1_turn = {word: i for i, word in enumerate(test_list1)}
#
# print( test_list1 )
# print( test_list1_turn )
#
# train_file = "test.txt"
# with open(train_file, "r", encoding="utf-8") as f:
#     for line in f:
#         print( line.strip().split("\t") )
#
# W = np.random.randn(3, 6) * 0.01
# print( W )
# W_scaled = W * np.sqrt(64)
# print( W_scaled )
#
# PE = np.zeros( (3, 6) )
# for pos in range(3):
#     for i in range(0, 6, 2):
#         angle = pos / (10000 ** (i / 6))
#
#         PE[pos, i] = np.sin(angle)
#
#         if i + 1 < 6:
#             PE[pos, i + 1] = np.cos(angle)
# print( PE )

import numpy as np

# x = np.array([
#     [1, 2, 3],
#     [4, 5, 6]
# ])
#
# print("原矩阵：")
# print(x)
# print("shape:", x.shape)
#
# # ===== axis=0 =====
# print("\naxis=0（跨行）最大值：")
# print(np.max(x, axis=0))
#
# # ===== axis=1 =====
# print("\naxis=1（跨列）最大值：")
# print(np.max(x, axis=1))

# x = np.random.randint(0, 10, (2, 3))
#
# print( x )
# #
# # print("shape:", x.shape)
#
# print("\n沿 axis=-1 最大值：")
# print(np.max(x, axis=-1, keepdims=True))
# # print("Ture : ")
# # print( x )
# print("\n沿 axis=-1 最大值：")
# print(np.max(x, axis=-1, keepdims=False))
# # print( "False : ")
# # print( x )
# print("做减法")
# print( x - np.max(x, axis=-1, keepdims=True))
# #print( x - np.max(x, axis=-1, keepdims=False) )

# import numpy as np
#
# A = np.array([[1, 2, 3, 4],
#               [5, 6, 7, 8]])
#
# N = A.shape[0]
# print( N )
#
# A = A.reshape(N, 2, 2)
# print( A )
#
# A = A.reshape(N, 4)
# print( A )

# x = np.array([[-0.1,2,3],[4,0.5,6],[-7,8,0.9]])
# print( np.maximum(0, x) )

# W = np.array([
#     [ 1,  2,  3,  4],   # token 0
#     [ 5,  6,  7,  8],   # token 1
#     [ 9, 10, 11, 12],   # token 2
#     [13, 14, 15, 16],   # token 3
#     [17, 18, 19, 20],   # token 4
#     [21, 22, 23, 24],   # token 5
# ])
#
# # 单句
# ids1 = np.array([1, 2, 3])
# print(W[ids1])   # (3, 4)
#
# # batch
# ids2 = np.array([
#     [1, 2, 3],
#     [4, 5, 1]
# ])
# print(W[ids2])   # (2, 3, 4)

# x = np.array([[1,1,1], [2,2,2]])
# y = np.array([3,3,3])
# print(x)
# print("after add")
# x = x + 5
# print(x)

import numpy as np

import json

def main():
    x = np.random.rand(6)
    print(x)
    mask = (x < 0.9).astype(np.float32)
    print(mask)


if __name__ == "__main__":
    main()