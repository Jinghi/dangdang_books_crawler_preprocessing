import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def show_preprocessing_results():
    """展示数据预处理结果"""
    print("=" * 60)
    print("当当网数据预处理结果展示")
    print("=" * 60)
    # 读取原始数据和处理后数据
    try:
        original_df = pd.read_csv(
            'dangdang_books.csv', encoding='utf-8-sig')
        processed_df = pd.read_csv(
            'dangdang_books_processed.csv', encoding='utf-8-sig')
        print(f"\n1. 数据规模对比:")
        print(f"   原始数据: {original_df.shape[0]} 行 × "
              f"{original_df.shape[1]} 列")
        print(f"   处理后数据: {processed_df.shape[0]} 行 × "
              f"{processed_df.shape[1]} 列")
        print(f"   数据减少: {original_df.shape[0] - processed_df.shape[0]} 行")
        print(f"   新增特征: {processed_df.shape[1] - original_df.shape[1]} 列")
        print(f"\n2. 数据预处理方法总结:")
        print(f"   ✓ 缺失值处理: 用中位数填充数值型，用'未知'填充文本型")
        print(f"   ✓ 数据去重: 基于书名和作者去重，删除 "
              f"{original_df.shape[0] - processed_df.shape[0]} 条重复数据")
        print(f"   ✓ 文本清洗: 移除敏感词、特殊字符，清理多余空格")
        print(f"   ✓ 异常值处理: 使用IQR方法处理价格和评论数异常值")
        print(f"   ✓ 数据转换: 统一数据类型，提取数字信息")
        print(f"   ✓ 敏感词过滤: 过滤包含营销词汇的数据")
        print(f"   ✓ 数据统计: 生成基本统计信息")
        print(f"   ✓ 特征工程: 新增价格区间和评分等级分类")
        print(f"\n3. 数据质量改进:")
        print(f"   - 缺失值: 从 {original_df.isnull().sum().sum()} 个减少到 "
              f"{processed_df.isnull().sum().sum()} 个")
        print(f"   - 数据一致性: 统一了价格、评分、评论数的格式")
        print(f"   - 数据完整性: 所有字段都有有效值")
        
        print(f"\n4. 新增特征说明:")
        print(f"   - 价格区间: 低价(0-20)、中价(20-50)、高价(50-100)、超高价(100+)")
        print(f"   - 评分等级: 差评(0-60)、一般(60-80)、好评(80-90)、优秀(90-100)")
        # 新增分箱特征说明
        if '价格_等宽分箱' in processed_df.columns:
            print(f"   - 价格_等宽分箱: 4等宽区间")
        if '价格_等频分箱' in processed_df.columns:
            print(f"   - 价格_等频分箱: 4等频区间")
        if '评分_等宽分箱' in processed_df.columns:
            print(f"   - 评分_等宽分箱: 4等宽区间")
        if '评分_等频分箱' in processed_df.columns:
            print(f"   - 评分_等频分箱: 4等频区间")
        if '评论数_等宽分箱' in processed_df.columns:
            print(f"   - 评论数_等宽分箱: 4等宽区间")
        if '评论数_等频分箱' in processed_df.columns:
            print(f"   - 评论数_等频分箱: 4等频区间")
        if 'PCA1' in processed_df.columns and 'PCA2' in processed_df.columns:
            print(f"   - PCA1/PCA2: 主成分分析降维特征")
        if '价格_评分' in processed_df.columns:
            print(f"   - 价格_评分: 价格与评分的交互特征")
        if '价格_评论数' in processed_df.columns:
            print(f"   - 价格_评论数: 价格与评论数的交互特征")
        if '评分_评论数' in processed_df.columns:
            print(f"   - 评分_评论数: 评分与评论数的交互特征")
        # 标准化/归一化
        for col in ['价格', '原价', '折扣', '评论数', '评分']:
            if f'{col}_标准化' in processed_df.columns:
                print(f"   - {col}_标准化: 标准化特征")
            if f'{col}_归一化' in processed_df.columns:
                print(f"   - {col}_归一化: 归一化特征")

        print(f"\n5. 数据统计摘要:")
        # 原有统计
        if '价格' in processed_df.columns:
            print(f"   - 价格范围: {processed_df['价格'].min():.2f} - {processed_df['价格'].max():.2f}")
            print(f"   - 平均价格: {processed_df['价格'].mean():.2f}")
            print(f"   - 价格中位数: {processed_df['价格'].median():.2f}")
        if '评分' in processed_df.columns:
            print(f"   - 评分范围: {processed_df['评分'].min():.2f} - {processed_df['评分'].max():.2f}")
            print(f"   - 平均评分: {processed_df['评分'].mean():.2f}")
        if '评论数' in processed_df.columns:
            print(f"   - 评论数范围: {processed_df['评论数'].min():.0f} - {processed_df['评论数'].max():.0f}")
            print(f"   - 平均评论数: {processed_df['评论数'].mean():.2f}")
        # 新增特征统计
        for col in ['PCA1', 'PCA2', '价格_评分', '价格_评论数', '评分_评论数']:
            if col in processed_df.columns:
                print(f"   - {col}：均值={processed_df[col].mean():.2f}，中位数={processed_df[col].median():.2f}")
        for col in ['价格', '原价', '折扣', '评论数', '评分']:
            if f'{col}_标准化' in processed_df.columns:
                print(f"   - {col}_标准化：均值={processed_df[f'{col}_标准化'].mean():.2f}，标准差={processed_df[f'{col}_标准化'].std():.2f}")
            if f'{col}_归一化' in processed_df.columns:
                print(f"   - {col}_归一化：最小值={processed_df[f'{col}_归一化'].min():.2f}，最大值={processed_df[f'{col}_归一化'].max():.2f}")

        print(f"\n6. 分类统计:")
        # 原有分箱统计
        if '价格区间' in processed_df.columns:
            print(f"   价格区间分布:")
            price_dist = processed_df['价格区间'].value_counts()
            for category, count in price_dist.items():
                percentage = (count / len(processed_df)) * 100
                print(f"     {category}: {count} 本 ({percentage:.1f}%)")
        if '评分等级' in processed_df.columns:
            print(f"   评分等级分布:")
            rating_dist = processed_df['评分等级'].value_counts()
            for category, count in rating_dist.items():
                percentage = (count / len(processed_df)) * 100
                print(f"     {category}: {count} 本 ({percentage:.1f}%)")
        # 新增分箱特征统计
        for col in ['价格_等宽分箱', '价格_等频分箱', '评分_等宽分箱', '评分_等频分箱', '评论数_等宽分箱', '评论数_等频分箱']:
            if col in processed_df.columns:
                print(f"   {col} 分布:")
                dist = processed_df[col].value_counts()
                for category, count in dist.items():
                    percentage = (count / len(processed_df)) * 100
                    print(f"     {category}: {count} 本 ({percentage:.1f}%)")
        
        print(f"\n7. 出版社TOP10:")
        if '出版社' in processed_df.columns:
            publisher_top10 = processed_df['出版社'].value_counts().head(10)
            for i, (publisher, count) in enumerate(publisher_top10.items(), 1):
                print(f"   {i:2d}. {publisher}: {count} 本")
        
        print(f"\n8. 数据预处理效果:")
        print(f"   ✓ 数据质量显著提升")
        print(f"   ✓ 数据格式统一规范")
        print(f"   ✓ 异常值和重复数据得到有效处理")
        print(f"   ✓ 新增分类特征便于分析")
        print(f"   ✓ 数据可用于后续的机器学习分析")
        
        # 创建可视化图表
        create_visualization_summary(processed_df)
        
    except FileNotFoundError as e:
        print(f"错误: 找不到数据文件 - {e}")
        print("请确保已运行 data_preprocessing.py 生成处理后的数据文件")

def create_visualization_summary(df):
    """创建数据可视化总结"""
    
    print(f"\n9. 生成数据可视化图表...")
    
    # 创建2x2的子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. 价格分布直方图
    if '价格' in df.columns:
        axes[0, 0].hist(df['价格'].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('价格分布', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('价格 (元)', fontsize=12)
        axes[0, 0].set_ylabel('频次', fontsize=12)
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 评分分布直方图
    if '评分' in df.columns:
        axes[0, 1].hist(df['评分'].dropna(), bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('评分分布', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('评分', fontsize=12)
        axes[0, 1].set_ylabel('频次', fontsize=12)
        axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 价格区间分布饼图
    if '价格区间' in df.columns:
        price_counts = df['价格区间'].value_counts()
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        axes[1, 0].pie(price_counts.values, labels=price_counts.index, autopct='%1.1f%%', 
                      colors=colors, startangle=90)
        axes[1, 0].set_title('价格区间分布', fontsize=14, fontweight='bold')
    
    # 4. 评分等级分布条形图
    if '评分等级' in df.columns:
        rating_counts = df['评分等级'].value_counts()
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        bars = axes[1, 1].bar(range(len(rating_counts)), rating_counts.values, color=colors)
        axes[1, 1].set_title('评分等级分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('评分等级', fontsize=12)
        axes[1, 1].set_ylabel('数量', fontsize=12)
        axes[1, 1].set_xticks(range(len(rating_counts)))
        axes[1, 1].set_xticklabels(rating_counts.index, rotation=45, ha='right')
        
        # 在柱状图上添加数值标签
        for bar, count in zip(bars, rating_counts.values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 10,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('preprocessing_results_summary.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"   可视化图表已保存为 'preprocessing_results_summary.png'")

if __name__ == "__main__":
    show_preprocessing_results() 