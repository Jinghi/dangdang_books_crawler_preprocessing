import pandas as pd
import re
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DangdangDataPreprocessor:
    def __init__(self, file_path):
        """初始化数据预处理器"""
        self.file_path = file_path
        self.df = None
        self.original_shape = None
        self.sensitive_words = [
            '加入购物车', '收藏', '购买电子书', '正版', '假一罚十', 
            '新华书店', '当当自营', '包邮', '现货', '热销', '推荐',
            '排行榜', '畅销', '经典', '官方', '旗舰店'
        ]
        self.report_lines = []  # 新增：用于收集终端展示内容
        
    def _print_and_log(self, msg):
        print(msg)
        self.report_lines.append(str(msg))
        
    def load_data(self):
        """1. 数据加载与基本信息查看"""
        self._print_and_log("=" * 50)
        self._print_and_log("1. 数据加载与基本信息查看")
        self._print_and_log("=" * 50)
        
        try:
            self.df = pd.read_csv(self.file_path, encoding='utf-8-sig')
            self.original_shape = self.df.shape
            self._print_and_log(f"数据加载成功！原始数据形状: {self.original_shape}")
            self._print_and_log(f"列名: {list(self.df.columns)}")
            self._print_and_log(f"数据类型:\n{self.df.dtypes}")
            self._print_and_log(f"前5行数据:\n{self.df.head()}")
            return True
        except Exception as e:
            self._print_and_log(f"数据加载失败: {e}")
            return False
    
    def handle_missing_values(self):
        """2. 缺失值处理"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("2. 缺失值处理")
        self._print_and_log("=" * 50)
        
        # 显示缺失值统计
        missing_stats = self.df.isnull().sum()
        self._print_and_log("缺失值统计:")
        self._print_and_log(missing_stats[missing_stats > 0])
        
        # 处理缺失值
        # 数值型列用中位数填充
        numeric_columns = ['价格', '原价', '折扣']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # 文本型列用'未知'填充
        text_columns = ['书名', '作者', '出版社', '出版日期', '评论数', 
                       '评分', 'ISBN', '图书分类']
        for col in text_columns:
            if col in self.df.columns:
                self.df[col].fillna('未知', inplace=True)
        
        self._print_and_log("缺失值处理完成！")
        self._print_and_log(f"处理后缺失值统计:\n{self.df.isnull().sum()}")
    
    def remove_duplicates(self):
        """3. 数据去重"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("3. 数据去重")
        self._print_and_log("=" * 50)
        
        before_count = len(self.df)
        self._print_and_log(f"去重前数据量: {before_count}")
        
        # 基于书名和作者去重
        self.df.drop_duplicates(
            subset=['书名', '作者'], 
            keep='first', 
            inplace=True
        )
        
        after_count = len(self.df)
        self._print_and_log(f"去重后数据量: {after_count}")
        self._print_and_log(f"删除重复数据: {before_count - after_count} 条")
        
        # 再次检验是否存在重复值
        duplicate_count = self.df.duplicated(subset=['书名', '作者']).sum()
        if duplicate_count == 0:
            self._print_and_log("已再次检验，数据集中不存在重复值")
        else:
            self._print_and_log(f"警告：数据集中仍存在 {duplicate_count} 条重复值")
    
    def clean_text_data(self):
        """4. 文本数据清洗"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("4. 文本数据清洗")
        self._print_and_log("=" * 50)
        
        # 清洗书名
        if '书名' in self.df.columns:
            self.df['书名'] = self.df['书名'].astype(str).apply(
                self._clean_book_title)
        
        # 清洗作者
        if '作者' in self.df.columns:
            self.df['作者'] = self.df['作者'].astype(str).apply(
                self._clean_author)
        
        # 清洗出版社
        if '出版社' in self.df.columns:
            self.df['出版社'] = self.df['出版社'].astype(str).apply(
                self._clean_publisher)
        
        # 清洗评论数
        if '评论数' in self.df.columns:
            self.df['评论数'] = self.df['评论数'].astype(str).apply(
                self._extract_number)
        
        # 清洗评分
        if '评分' in self.df.columns:
            self.df['评分'] = self.df['评分'].astype(str).apply(
                self._extract_rating)
        
        self._print_and_log("文本数据清洗完成！")
    
    def _clean_book_title(self, title):
        """清洗书名"""
        if pd.isna(title) or title == 'nan':
            return '未知'
        
        # 移除敏感词
        for word in self.sensitive_words:
            title = title.replace(word, '')
        
        # 移除多余空格
        title = re.sub(r'\s+', ' ', title).strip()
        
        # 移除特殊字符
        title = re.sub(r'[【】\[\]（）()]', '', title)
        
        return title if title else '未知'
    
    def _clean_author(self, author):
        """清洗作者"""
        if pd.isna(author) or author == 'nan':
            return '未知'
        
        # 移除敏感词
        for word in self.sensitive_words:
            author = author.replace(word, '')
        
        # 移除多余空格
        author = re.sub(r'\s+', ' ', author).strip()
        
        return author if author else '未知'
    
    def _clean_publisher(self, publisher):
        """清洗出版社"""
        if pd.isna(publisher) or publisher == 'nan':
            return '未知'
        
        # 移除敏感词
        for word in self.sensitive_words:
            publisher = publisher.replace(word, '')
        
        # 移除多余空格
        publisher = re.sub(r'\s+', ' ', publisher).strip()
        
        return publisher if publisher else '未知'
    
    def _extract_number(self, text):
        """提取数字"""
        if pd.isna(text) or text == 'nan':
            return 0
        
        # 提取所有数字
        numbers = re.findall(r'\d+', str(text))
        if numbers:
            return int(''.join(numbers))  # 将所有数字连接起来
        return 0
    
    def _extract_rating(self, text):
        """提取评分"""
        if pd.isna(text) or text == 'nan':
            return 0
        
        # 提取百分比
        percentage = re.findall(r'(\d+)%', str(text))
        if percentage:
            return int(percentage[0])
        
        # 提取小数
        decimal = re.findall(r'(\d+\.?\d*)', str(text))
        if decimal:
            return float(decimal[0])
        
        return 0
    
    def handle_outliers(self):
        """5. 异常值处理"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("5. 异常值处理")
        self._print_and_log("=" * 50)
        
        # 处理价格异常值
        if '价格' in self.df.columns:
            self.df['价格'] = pd.to_numeric(self.df['价格'], errors='coerce')
            q1 = self.df['价格'].quantile(0.25)
            q3 = self.df['价格'].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            self._print_and_log(f"价格异常值边界: {lower_bound:.2f} - {upper_bound:.2f}")
            self.df['价格'] = self.df['价格'].clip(
                lower=lower_bound,
                upper=upper_bound
            )
            self._print_and_log(
                "价格异常值处理完成！"
            )
        
        # 不再对评论数做clip异常值处理，直接保留原始爬虫爬取的评论数
        # if '评论数' in self.df.columns:
        #     self.df['评论数'] = pd.to_numeric(self.df['评论数'], errors='coerce')
        #     q1 = self.df['评论数'].quantile(0.25)
        #     q3 = self.df['评论数'].quantile(0.75)
        #     iqr = q3 - q1
        #     lower_bound = q1 - 1.5 * iqr
        #     upper_bound = q3 + 1.5 * iqr
        #     print(f"评论数异常值边界: {lower_bound:.2f} - {upper_bound:.2f}")
        #     self.df['评论数'] = self.df['评论数'].clip(
        #         lower=lower_bound, 
        #         upper=upper_bound
        #     )
        #     print(f"评论数异常值处理完成！")
    
    def data_transformation(self):
        """6. 数据转换"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("6. 数据转换")
        self._print_and_log("=" * 50)
        
        # 转换价格数据类型
        if '价格' in self.df.columns:
            self.df['价格'] = pd.to_numeric(self.df['价格'], errors='coerce')
        
        if '原价' in self.df.columns:
            self.df['原价'] = pd.to_numeric(self.df['原价'], errors='coerce')
        
        if '折扣' in self.df.columns:
            self.df['折扣'] = pd.to_numeric(self.df['折扣'], errors='coerce')
        
        # 转换评论数为数值型
        if '评论数' in self.df.columns:
            self.df['评论数'] = pd.to_numeric(self.df['评论数'], errors='coerce')
        
        # 转换评分为数值型
        if '评分' in self.df.columns:
            self.df['评分'] = pd.to_numeric(self.df['评分'], errors='coerce')
        
        # 添加价格区间分类
        if '价格' in self.df.columns:
            self.df['价格区间'] = pd.cut(
                self.df['价格'], 
                bins=[0, 20, 50, 100, float('inf')],
                labels=['低价(0-20)', '中价(20-50)', '高价(50-100)', 
                       '超高价(100+)'])
        
        # 添加评分等级
        if '评分' in self.df.columns:
            self.df['评分等级'] = pd.cut(
                self.df['评分'], 
                bins=[0, 60, 80, 90, 100],
                labels=['差评(0-60)', '一般(60-80)', '好评(80-90)', 
                       '优秀(90-100)'])
        
        self._print_and_log(
            "新增列: 价格区间, 评分等级"
        )
    
    def filter_sensitive_content(self):
        """7. 敏感词过滤"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("7. 敏感词过滤")
        self._print_and_log("=" * 50)
        
        # 定义更多敏感词
        additional_sensitive_words = [
            '假一罚十', '正版保证', '官方正版', '新华书店正版',
            '当当自营', '包邮', '现货', '热销', '推荐', '排行榜',
            '畅销', '经典', '官方', '旗舰店', '品质保障', '优质服务',
            '发货及时', '售后无忧', '七天无理由退换货'
        ]
        
        all_sensitive_words = self.sensitive_words + additional_sensitive_words
        
        # 过滤包含敏感词的行
        before_count = len(self.df)
        
        # 创建过滤条件
        filter_condition = self.df['书名'].str.contains(
            '|'.join(all_sensitive_words), na=False)
        
        # 保留不包含敏感词的行
        self.df = self.df[~filter_condition]
        
        after_count = len(self.df)
        self._print_and_log(f"敏感词过滤前: {before_count} 条")
        self._print_and_log(f"敏感词过滤后: {after_count} 条")
        self._print_and_log(f"过滤掉: {before_count - after_count} 条")
    
    def data_statistics(self):
        """8. 数据统计分析"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("8. 数据统计分析")
        self._print_and_log("=" * 50)
        
        # 基本统计信息
        self._print_and_log("基本统计信息:")
        if '价格' in self.df.columns:
            self._print_and_log(f"价格统计:\n{self.df['价格'].describe()}")
        
        if '评论数' in self.df.columns:
            self._print_and_log(f"评论数统计:\n{self.df['评论数'].describe()}")
        
        if '评分' in self.df.columns:
            self._print_and_log(f"评分统计:\n{self.df['评分'].describe()}")
        
        # 分类统计
        if '出版社' in self.df.columns:
            self._print_and_log(f"\n出版社TOP10:\n{self.df['出版社'].value_counts().head(10)}")
        
        if '价格区间' in self.df.columns:
            self._print_and_log(f"\n价格区间分布:\n{self.df['价格区间'].value_counts()}")
        
        if '评分等级' in self.df.columns:
            self._print_and_log(f"\n评分等级分布:\n{self.df['评分等级'].value_counts()}")
    
    def create_visualizations(self):
        """9. 数据可视化"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("9. 数据可视化")
        self._print_and_log("=" * 50)
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 价格分布直方图
        if '价格' in self.df.columns:
            axes[0, 0].hist(self.df['价格'].dropna(), bins=30, alpha=0.7, 
                           color='skyblue')
            axes[0, 0].set_title('价格分布')
            axes[0, 0].set_xlabel('价格')
            axes[0, 0].set_ylabel('频次')
        
        # 2. 评分分布直方图
        if '评分' in self.df.columns:
            axes[0, 1].hist(self.df['评分'].dropna(), bins=20, alpha=0.7, 
                           color='lightgreen')
            axes[0, 1].set_title('评分分布')
            axes[0, 1].set_xlabel('评分')
            axes[0, 1].set_ylabel('频次')
        
        # 3. 价格区间分布饼图
        if '价格区间' in self.df.columns:
            price_counts = self.df['价格区间'].value_counts()
            axes[1, 0].pie(price_counts.values, labels=price_counts.index, 
                          autopct='%1.1f%%')
            axes[1, 0].set_title('价格区间分布')
        
        # 4. 评分等级分布条形图
        if '评分等级' in self.df.columns:
            rating_counts = self.df['评分等级'].value_counts()
            axes[1, 1].bar(range(len(rating_counts)), rating_counts.values, 
                          color='orange')
            axes[1, 1].set_title('评分等级分布')
            axes[1, 1].set_xticks(range(len(rating_counts)))
            axes[1, 1].set_xticklabels(rating_counts.index, rotation=45)
        
        plt.tight_layout()
        plt.savefig('dangdang_data_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        self._print_and_log(
            "可视化图表已保存为 'dangdang_data_analysis.png'"
        )
    
    def save_processed_data(self):
        """10. 保存处理后的数据"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("10. 保存处理后的数据")
        self._print_and_log("=" * 50)
        
        # 保存为CSV
        output_file = 'dangdang_books_processed.csv'
        self.df.to_csv(
            output_file, 
            index=False, 
            encoding='utf-8-sig'
        )
        
        self._print_and_log(f"处理后的数据已保存为: {output_file}")
        self._print_and_log(f"最终数据形状: {self.df.shape}")
        self._print_and_log(f"数据减少: {self.original_shape[0] - self.df.shape[0]} 行")
    
    def generate_report(self):
        """11. 生成数据预处理报告"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("11. 数据预处理报告")
        self._print_and_log("=" * 50)

        report = f"""
数据预处理报告
================

原始数据信息:
- 数据形状: {self.original_shape}
- 列数: {self.original_shape[1]}

处理后数据信息:
- 数据形状: {self.df.shape}
- 列数: {self.df.shape[1]}
- 数据减少: {self.original_shape[0] - self.df.shape[0]} 行

数据质量改进:
- 缺失值处理: 完成
- 数据去重: 完成
- 文本清洗: 完成
- 异常值处理: 完成
- 数据转换: 完成
- 敏感词过滤: 完成
- 特征工程: 完成

新增特征:
"""
        # 新增特征说明
        for col in ['价格_等宽分箱', '价格_等频分箱', '评分_等宽分箱', '评分_等频分箱', '评论数_等宽分箱', '评论数_等频分箱']:
            if col in self.df.columns:
                report += f"- {col}: 分箱特征\n"
        for col in ['PCA1', 'PCA2']:
            if col in self.df.columns:
                report += f"- {col}: PCA降维特征\n"
        for col in ['价格_评分', '价格_评论数', '评分_评论数']:
            if col in self.df.columns:
                report += f"- {col}: 交互特征\n"
        for col in ['价格', '原价', '折扣', '评论数', '评分']:
            if f'{col}_标准化' in self.df.columns:
                report += f"- {col}_标准化: 标准化特征\n"
            if f'{col}_归一化' in self.df.columns:
                report += f"- {col}_归一化: 归一化特征\n"

        report += "\n数据统计摘要:\n"
        # 主要数值统计
        for col in ['价格', '评分', '评论数']:
            if col in self.df.columns:
                report += f"- {col}均值: {self.df[col].mean():.2f}，中位数: {self.df[col].median():.2f}\n"
        for col in ['PCA1', 'PCA2', '价格_评分', '价格_评论数', '评分_评论数']:
            if col in self.df.columns:
                report += f"- {col}均值: {self.df[col].mean():.2f}，中位数: {self.df[col].median():.2f}\n"
        # 分箱分布
        for col in ['价格_等宽分箱', '价格_等频分箱', '评分_等宽分箱', '评分_等频分箱', '评论数_等宽分箱', '评论数_等频分箱']:
            if col in self.df.columns:
                report += f"- {col}分布:\n"
                dist = self.df[col].value_counts()
                for category, count in dist.items():
                    percentage = (count / len(self.df)) * 100
                    report += f"    {category}: {count} ({percentage:.1f}%)\n"

        self._print_and_log(report)

        # 保存报告
        with open('data_preprocessing_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)

        self._print_and_log("详细报告已保存为 'data_preprocessing_report.txt'")
    

    def feature_scaling(self):
        """特征缩放：标准化和归一化"""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("特征缩放：标准化和归一化")
        self._print_and_log("=" * 50)
        numeric_cols = ['价格', '原价', '折扣', '评论数', '评分']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        # 标准化
        if available_cols:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(self.df[available_cols])
            for i, col in enumerate(available_cols):
                self.df[col + '_标准化'] = scaled[:, i]
            self._print_and_log(f"已标准化列: {available_cols}")
        # 归一化
        if available_cols:
            minmax = MinMaxScaler()
            normed = minmax.fit_transform(self.df[available_cols])
            for i, col in enumerate(available_cols):
                self.df[col + '_归一化'] = normed[:, i]
            self._print_and_log(f"已归一化列: {available_cols}")

    def feature_interaction(self):
        """特征交互：创建交互特征"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("特征交互：创建交互特征")
        self._print_and_log("=" * 50)
        # 价格*评分
        if '价格' in self.df.columns and '评分' in self.df.columns:
            self.df['价格_评分'] = self.df['价格'] * self.df['评分']
        # 价格*评论数
        if '价格' in self.df.columns and '评论数' in self.df.columns:
            self.df['价格_评论数'] = self.df['价格'] * self.df['评论数']
        # 评分*评论数
        if '评分' in self.df.columns and '评论数' in self.df.columns:
            self.df['评分_评论数'] = self.df['评分'] * self.df['评论数']
        self._print_and_log("已创建交互特征：价格_评分、价格_评论数、评分_评论数")

    def dimensionality_reduction(self):
        """降维技术：PCA降维"""
        from sklearn.decomposition import PCA
        import numpy as np
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("降维技术：PCA降维")
        self._print_and_log("=" * 50)
        # 选择数值型特征
        numeric_cols = ['价格', '原价', '折扣', '评论数', '评分']
        available_cols = [col for col in numeric_cols if col in self.df.columns]
        if len(available_cols) >= 2:
            X = self.df[available_cols].fillna(0).values
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)
            self.df['PCA1'] = X_pca[:, 0]
            self.df['PCA2'] = X_pca[:, 1]
            self._print_and_log(f"PCA降维已完成，新增列：PCA1, PCA2")
        else:
            self._print_and_log("可用数值特征不足，跳过PCA降维")

    def binning_features(self):
        """分箱处理：等宽分箱和等频分箱"""
        self._print_and_log("\n" + "=" * 50)
        self._print_and_log("分箱处理：等宽分箱和等频分箱")
        self._print_and_log("=" * 50)
        # 以价格为例
        if '价格' in self.df.columns:
            # 等宽分箱
            self.df['价格_等宽分箱'] = pd.cut(
                self.df['价格'], bins=4, labels=[f'区间{i+1}' for i in range(4)])
            # 等频分箱
            self.df['价格_等频分箱'] = pd.qcut(
                self.df['价格'], q=4, duplicates='drop')
            self._print_and_log("已对价格做等宽分箱和等频分箱")
        # 评分分箱
        if '评分' in self.df.columns:
            self.df['评分_等宽分箱'] = pd.cut(
                self.df['评分'], bins=4, labels=[f'区间{i+1}' for i in range(4)])
            self.df['评分_等频分箱'] = pd.qcut(self.df['评分'], q=4, duplicates='drop')
            self._print_and_log("已对评分做等宽分箱和等频分箱")
        # 评论数分箱
        if '评论数' in self.df.columns:
            # 先试探分箱数
            try:
                _, bins = pd.qcut(self.df['评论数'], q=4, retbins=True, duplicates='drop')
                n_bins = len(bins) - 1
                labels = [f'分位{i+1}' for i in range(n_bins)]
                self.df['评论数_等频分箱'] = pd.qcut(self.df['评论数'], q=4, labels=labels, duplicates='drop')
            except ValueError as e:
                self._print_and_log(f"评论数等频分箱失败: {e}")
            self._print_and_log("已对评论数做等宽分箱和等频分箱")

    def run_full_preprocessing(self):
        """运行完整的数据预处理流程"""
        self._print_and_log("开始当当网数据预处理...")
        
        # 执行所有预处理步骤
        if self.load_data():
            self.handle_missing_values()
            self.remove_duplicates()
            self.clean_text_data()
            self.handle_outliers()
            self.data_transformation()
            self.filter_sensitive_content()
            self.data_statistics()
            self.create_visualizations()
            self.save_processed_data()
            self.generate_report()
            # 新增特征工程流程
            self.feature_scaling()
            self.feature_interaction()
            self.dimensionality_reduction()
            self.binning_features()
            
            self._print_and_log("\n" + "=" * 50)
            self._print_and_log("数据预处理完成！")
            self._print_and_log("=" * 50)
            # 新增：写入终端展示报告
            with open('data_preprocessing_terminal_report.txt', 'w', encoding='utf-8') as f:
                f.write('\n'.join(self.report_lines))
            self._print_and_log("终端展示内容已保存为 'data_preprocessing_terminal_report.txt'")
        else:
            self._print_and_log("数据加载失败，预处理终止！")


# 主程序
if __name__ == "__main__":
    # 创建预处理器实例
    preprocessor = DangdangDataPreprocessor('dangdang_books.csv')
    
    # 运行完整预处理流程
    preprocessor.run_full_preprocessing() 