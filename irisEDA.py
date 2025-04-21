import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = sns.load_dataset('iris')

# Display first few rows
print(iris.head())

plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal_length', y='petal_length', 
                hue='species', style='species',
                s=100, alpha=0.8, data=iris)
plt.title('Sepal Length vs Petal Length by Iris Species', pad=20)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

correlation = iris['sepal_length'].corr(iris['petal_length'])
print(f"Pearson correlation coefficient between sepal length and petal length: {correlation:.3f}")

g = sns.lmplot(x='sepal_length', y='petal_length', col='species', 
               hue='species', data=iris, height=5, aspect=0.8)
g.fig.suptitle('Sepal vs Petal Length by Species', y=1.05)
plt.show()