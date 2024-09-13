import matplotlib.pyplot as plt

# 단순 그래프를 그려봅니다
plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Test Plot')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.savefig('test_plot.png')  # 파일로 저장
plt.show()  # 창으로 표시
