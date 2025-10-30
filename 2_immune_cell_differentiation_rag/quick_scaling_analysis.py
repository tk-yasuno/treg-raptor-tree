"""
Quick Scaling Analysis
既に得られた結果を分析
"""

import matplotlib.pyplot as plt
import numpy as np

def analyze_scaling_results():
    """スケーリング結果を分析"""
    
    print("🧬 Immune Cell RAPTOR Tree Scaling Analysis")
    print("=" * 50)
    
    # 実測結果
    scale_1x = {
        "scale_factor": 1,
        "articles_per_query": 15,
        "total_articles": 133,
        "time": 50.75
    }
    
    scale_2x = {
        "scale_factor": 2,
        "articles_per_query": 30,
        "total_articles": 248,
        "time": 66.20
    }
    
    # スケーリング分析
    time_ratio = scale_2x["time"] / scale_1x["time"]
    articles_ratio = scale_2x["total_articles"] / scale_1x["total_articles"]
    
    print("📊 SCALING RESULTS:")
    print(f"Base (1x): {scale_1x['time']:.2f} seconds, {scale_1x['total_articles']} articles")
    print(f"Scale (2x): {scale_2x['time']:.2f} seconds, {scale_2x['total_articles']} articles")
    print(f"Time scaling ratio: {time_ratio:.2f}x")
    print(f"Articles scaling ratio: {articles_ratio:.2f}x")
    
    print("\\n🎯 ANALYSIS:")
    if time_ratio < 1.8:
        print("✅ EXCELLENT: Sub-linear time scaling!")
        print("💡 The system scales efficiently - no immediate need for parallelization")
        print(f"   Only {time_ratio:.1f}x time increase for {articles_ratio:.1f}x more data")
    else:
        print("⚠️ Near-linear scaling detected")
        print("💡 Parallel processing could provide benefits")
    
    # 効率分析
    efficiency_1x = scale_1x["total_articles"] / scale_1x["time"]
    efficiency_2x = scale_2x["total_articles"] / scale_2x["time"]
    
    print(f"\\n⚡ EFFICIENCY:")
    print(f"1x throughput: {efficiency_1x:.1f} articles/second")
    print(f"2x throughput: {efficiency_2x:.1f} articles/second")
    
    if efficiency_2x > efficiency_1x:
        print("🚀 IMPROVING: System gets more efficient at larger scales!")
    else:
        print("📉 DECREASING: System efficiency decreases with scale")
    
    # CPU コア使用推奨
    cpu_cores = 8  # Windows 環境
    potential_speedup = min(cpu_cores, 4)  # I/O bound タスクなので4倍程度が上限
    
    print(f"\\n🔧 PARALLELIZATION RECOMMENDATIONS:")
    print(f"Available CPU cores: {cpu_cores}")
    print(f"Potential speedup with parallel processing: ~{potential_speedup}x")
    
    if time_ratio < 1.5:
        print("📝 RECOMMENDATION: Current performance is excellent for this scale")
        print("   Consider parallel processing only for 4x+ scales or production use")
    else:
        print("📝 RECOMMENDATION: Implement parallel processing for better performance")
    
    # 可視化
    create_visualization(scale_1x, scale_2x, time_ratio, efficiency_1x, efficiency_2x)

def create_visualization(scale_1x, scale_2x, time_ratio, eff_1x, eff_2x):
    """結果可視化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. 実行時間比較
    scales = [1, 2]
    times = [scale_1x["time"], scale_2x["time"]]
    articles = [scale_1x["total_articles"], scale_2x["total_articles"]]
    
    ax1.bar(scales, times, alpha=0.7, color='skyblue', label='Actual')
    
    # 理想的なリニアスケーリング
    linear_times = [scale_1x["time"], scale_1x["time"] * 2]
    ax1.plot(scales, linear_times, 'r--', label='Linear scaling', linewidth=2)
    
    ax1.set_xlabel('Scale Factor')
    ax1.set_ylabel('Execution Time (seconds)')
    ax1.set_title('Scaling Performance')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 記事数とスケール
    ax2.bar(scales, articles, alpha=0.7, color='lightgreen')
    ax2.set_xlabel('Scale Factor')
    ax2.set_ylabel('Total Articles Retrieved')
    ax2.set_title('Articles vs Scale Factor')
    ax2.grid(True, alpha=0.3)
    
    # 3. 効率比較
    efficiencies = [eff_1x, eff_2x]
    bars = ax3.bar(scales, efficiencies, alpha=0.7, color='orange')
    ax3.set_xlabel('Scale Factor')
    ax3.set_ylabel('Articles/Second')
    ax3.set_title('Throughput Efficiency')
    ax3.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, eff in zip(bars, efficiencies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{eff:.1f}', ha='center', va='bottom')
    
    # 4. スケーリング係数
    ratios = ['Time Ratio', 'Articles Ratio']
    values = [time_ratio, articles[1]/articles[0]]
    colors = ['red' if time_ratio > 1.8 else 'green', 'blue']
    
    bars = ax4.bar(ratios, values, alpha=0.7, color=colors)
    ax4.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Linear (2x)')
    ax4.axhline(y=1.0, color='gray', linestyle='-', alpha=0.3)
    ax4.set_ylabel('Ratio')
    ax4.set_title('Scaling Ratios')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 値をバーの上に表示
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2f}x', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('scaling_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n📊 Visualization saved as: scaling_analysis_results.png")

if __name__ == "__main__":
    analyze_scaling_results()