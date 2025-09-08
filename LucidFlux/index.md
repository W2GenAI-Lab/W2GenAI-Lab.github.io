---
layout: default
permalink: /LucidFlux/
title: LucidFlux Universal Image Restoration with a Large-Scale Diffusion Transformer
extra_css:
  - /assets/css/lucidflux.css
extra_js:
  - /assets/js/lucidflux.js
---

# LucidFlux

LucidFlux 是我们先进的图像生成模型，利用流程匹配技术结合扩散模型来生成高分辨率、高质量的图像。

<!-- 顶部浮动demo图片 - 更新版本 -->
<div class="floating-demos">
    <!-- 第一组demo图片 -->
    <div class="demo-left">
        <img src="/LucidFlux/gallery/concatenated_202.jpg" alt="LucidFlux Generated Demo 1">
    </div>
    <div class="demo-right">
        <img src="/LucidFlux/gallery/concatenated_137.jpg" alt="LucidFlux Generated Demo 2">
    </div>
    
    <!-- 第二组demo图片 -->
    <div class="demo-second-left">
        <img src="/LucidFlux/gallery/concatenated_013.jpg" alt="LucidFlux Generated Demo 3">
    </div>
    <div class="demo-second-right">
        <img src="/LucidFlux/gallery/concatenated_079.jpg" alt="LucidFlux Generated Demo 4">
    </div>
</div>

<div class="hero">
    <div class="hero-content">
        <!-- 主标题区域：LucidFlux + Logo -->
        <div class="title-logo-section">
            <h1 class="hero-title">LucidFlux:</h1>
            <div class="inline-logo">
                <div class="logo-glow-effect-inline"></div>
                <img src="/LucidFlux/architecture/arch.png" alt="LucidFlux Logo" class="main-logo-inline">
                <div class="logo-sparkles-inline">
                    <span class="logo-sparkle logo-sparkle-1">✨</span>
                    <span class="logo-sparkle logo-sparkle-2">⭐</span>
                </div>
            </div>
        </div>
        
        <!-- Paper信息 - 紧凑布局 -->
        <div class="paper-info-compact">
            <h3 class="paper-title-main">
                Universal Image Restoration with a Large-Scale Diffusion Transformer
            </h3>
            <div class="paper-badge-fancy">📄 Research Paper</div>
        </div>
        
        <!-- 作者信息 -->
        <div class="authors-section">
            <p class="authors">
                <span class="author"><a href="https://ephemeral182.github.io/">Sixiang Chen</a><sup>1,2,*</sup></span>,
                <span class="author"><a href="https://openreview.net/profile?id=~Jianyu_Lai1">Jianyu Lai</a><sup>1,*</sup></span>,
                <span class="author"><a href="https://scholar.google.com/citations?user=sj4FqEgAAAAJ&hl=zh-CN">Jialin Gao</a><sup>2,*</sup></span>,
                <span class="author"><a href="https://owen718.github.io/">Tian Ye</a><sup>1</sup></span>,
                <span class="author"><a href="https://haoyuchen.com/">Haoyu Chen</a><sup>1</sup></span>,
                <span class="author"><a href="https://openreview.net/profile?id=%7EHengyu_Shi1">Hengyu Shi</a><sup>2</sup></span>,
                <span class="author"><a href="https://shaoshitong.github.io/">Shitong Shao</a><sup>1</sup></span>,
                <span class="author"><a href="https://scholar.google.com.hk/citations?user=5F3tICwAAAAJ&hl=zh-CN">Yunlong Lin</a><sup>3</sup></span>,
                <span class="author"><a href="https://openreview.net/profile?id=~Song_Fei1">Song Fei</a><sup>1</sup></span>,
                <span class="author"><a href="https://ge-xing.github.io/">Zhaohu Xing</a><sup>1</sup></span>,
                <span class="author"><a href="https://jinyeying.github.io/">Yeying Jin</a><sup>4</sup></span>,
                <span class="author">Junfeng Luo<sup>2</sup></span>,
                <span class="author"><a href="https://scholar.google.com/citations?user=JXV5yrZxj5MC&hl=zh-CN">Xiaoming Wei</a><sup>2</sup></span>,
                <span class="author"><a href="https://sites.google.com/site/indexlzhu/home">Lei Zhu</a><sup>1,5,†</sup></span>
            </p>
            <p class="affiliations">
                <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou),
                <sup>2</sup>Meituan,
                <sup>3</sup>Xiamen University,
                <sup>4</sup>National University of Singapore,
                <sup>5</sup>The Hong Kong University of Science and Technology
            </p>
            <p class="contributions">
                <sup>*</sup>Equal Contribution, <sup>†</sup>Corresponding Author
            </p>
        </div>
        
        <!-- 论文链接 - 增强版 -->
        <div class="paper-links-enhanced">
            <a href="https://arxiv.org/abs/2506.10741" class="paper-link-fancy arxiv-link" target="_blank">
                <div class="link-icon-fancy">📚</div>
                <span>arXiv</span>
                <div class="link-shine"></div>
            </a>
            <a href="https://github.com/Ephemeral182/PosterCraft" class="paper-link-fancy github-link" target="_blank">
                <div class="link-icon-fancy">💻</div>
                <span>GitHub</span>
                <div class="link-shine"></div>
            </a>
            <a href="https://huggingface.co/PosterCraft/PosterCraft-v1_RL" class="paper-link-fancy huggingface-link" target="_blank">
                <div class="link-icon-fancy">🤗</div>
                <span>HuggingFace</span>
                <div class="link-shine"></div>
            </a>
            <a href="https://huggingface.co/spaces/Ephemeral182/PosterCraft" class="paper-link-fancy demo-link" target="_blank">
                <div class="link-icon-fancy">🚀</div>
                <span>HF Demo</span>
                <div class="link-shine"></div>
            </a>
            <a href="https://huggingface.co/PosterCraft" class="paper-link-fancy dataset-link" target="_blank">
                <div class="link-icon-fancy">📊</div>
                <span>Datasets</span>
                <div class="link-shine"></div>
            </a>
            <a href="https://www.youtube.com/watch?v=92wMU4D7qx0" class="paper-link-fancy video-link" target="_blank">
                <div class="link-icon-fancy">🎥</div>
                <span>Demo Video</span>
                <div class="link-shine"></div>
            </a>
        </div>
    </div>
    <div class="hero-visual">
        <div class="mosaic-grid">
            <div class="mosaic-item mosaic-1">
                <img src="/LucidFlux/gallery/concatenated_214.jpg" alt="LucidFlux Generated 1">
            </div>
            <div class="mosaic-item mosaic-2">
                <img src="/LucidFlux/gallery/concatenated_183.jpg" alt="LucidFlux Generated 2">
            </div>
            <div class="mosaic-item mosaic-9">
                <img src="/LucidFlux/gallery/concatenated_144.jpg" alt="LucidFlux Generated 3">
            </div>
            <div class="mosaic-item mosaic-4">
                <img src="/LucidFlux/gallery/concatenated_231.jpg" alt="LucidFlux Generated 4">
            </div>
            <div class="mosaic-item mosaic-5">
                <img src="/LucidFlux/gallery/concatenated_041.jpg" alt="LucidFlux Generated 5">
            </div>
            <div class="mosaic-item mosaic-6">
                <img src="/LucidFlux/gallery/concatenated_202.jpg" alt="LucidFlux Generated 6">
            </div>
            <div class="mosaic-item mosaic-7">
                <img src="/LucidFlux/gallery/concatenated_137.jpg" alt="LucidFlux Generated 7">
            </div>
            <div class="mosaic-item mosaic-8">
                <img src="/LucidFlux/gallery/concatenated_013.jpg" alt="LucidFlux Generated 8">
            </div>
        </div>
    </div>
</div>

<!-- Video Showcase Section -->
<div class="video-showcase-section" id="video-showcase">
    <div class="video-showcase-container">
        <div class="video-header">
            <h2 class="video-title">Demo Video</h2>
        </div>
        
        <div class="video-main-container">
            <div class="video-frame-wrapper">
                <!-- 新片场嵌入播放器 (更新版) -->
                <iframe 
                    src="https://player.xinpianchang.com/?aid=13399853&amp;mid=0vMbw1zoN1D4eJ1p&autoplay=0&muted=0"
                    scrolling="no" 
                    border="0" 
                    frameborder="no" 
                    framespacing="0" 
                    allowfullscreen="true"
                    class="bilibili-video">
                </iframe>
                
                <div class="video-glow-effect"></div>
            </div>
        </div>
    </div>
</div>

<div class="what-is-section" id="what-is">
    <div class="what-is-container">
        <h2 class="what-is-title">What is LucidFlux?</h2>
        <div style="text-align: center;">
            <div class="what-is-emoji">🌊</div>
        </div>
        <p class="what-is-subtitle">From your prompts to high-resolution images, LucidFlux excels in flow matching technology, stable training, efficient memory usage, and superior image quality.</p>
        
        <!-- 新的横向布局 -->
        <div class="horizontal-demo-container">
            <!-- Quick Prompt Demo -->
            <div class="demo-row">
                <div class="demo-row-background">
                    <div class="demo-label-horizontal">
                        <span class="label-icon">⚡</span>
                        <span class="label-text">Quick Prompt</span>
                    </div>
                    <div class="horizontal-transformation">
                        <div class="input-section">
                            <div class="input-card-horizontal">
                                <div class="input-header">
                                    <span class="input-icon">✨</span>
                                    <span>Simple Description</span>
                                </div>
                                <div class="input-content">
                                    <p>"<span class="hl-detail">Urban Canvas Street Art Expo</span> poster with bold graffiti-style lettering and dynamic colorful splashes"</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="arrow-section">
                            <div class="magic-arrow-horizontal">
                                <span class="arrow-text">LucidFlux</span>
                                <div class="arrow-line-horizontal">
                                    <div class="arrow-head-horizontal"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="output-section">
                            <div class="output-frame-horizontal">
                                <img src="/LucidFlux/gallery/concatenated_079.jpg" alt="Beautiful Landscape Generated by LucidFlux">
                                <div class="output-glow-horizontal"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <!-- Detailed Prompt Demo -->
            <div class="demo-row">
                <div class="demo-row-background">
                    <div class="demo-label-horizontal">
                        <span class="label-icon">🎯</span>
                        <span class="label-text">Detailed Prompt</span>
                    </div>
                    <div class="horizontal-transformation">
                        <div class="input-section">
                            <div class="input-card-horizontal">
                                <div class="input-header">
                                    <span class="input-icon">🎨</span>
                                    <span>Detailed Description</span>
                                </div>
                                <div class="input-content">
                                    <p>"This poster for the post-apocalyptic film "<span class="hl-detail">Echoes of the Shattered Sun</span>" showcases a lone survivor in ragged clothes, standing on a desolate, cracked earth under a sky dominated by a fragmented, dying sun that casts long, eerie shadows. Ruined cityscapes are barely visible on the horizon. The style is bleak, atmospheric, and visually striking, emphasizing despair and a fight for survival. The film's title, "<span class="hl-detail">ECHOES OF THE SHATTERED SUN</span>" is presented in a fragmented, futuristic, sans-serif font, the letters appearing as if broken and pieced together from salvaged metal, with a faint, dying orange glow. This text is positioned horizontally across the top of the poster, large and ominous. Below the survivor, the release information "<span class="hl-detail">THE FUTURE IS BROKEN. SURVIVAL IS ALL THAT REMAINS. COMING SOON</span>" is in a smaller, gritty, white stencil font, horizontally centered. The fractured, thematic title amplifying the film's dystopian and survivalist themes."</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="arrow-section">
                            <div class="magic-arrow-horizontal">
                                <span class="arrow-text">LucidFlux</span>
                                <div class="arrow-line-horizontal">
                                    <div class="arrow-head-horizontal"></div>
                                </div>
                            </div>
                        </div>
                        
                        <div class="output-section">
                            <div class="output-frame-horizontal">
                                <img src="/LucidFlux/gallery/concatenated_214.jpg" alt="Artistic Portrait Generated by LucidFlux">
                                <div class="output-glow-horizontal"></div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>

<!-- Model Architecture Section -->
<div class="technical-overview-section" id="model-architecture">
    <div class="overview-container">
        <div class="overview-header">
            <h2 class="overview-title">LucidFlux Architecture</h2>
            <div class="overview-emoji">🏗️</div>
            <p class="overview-subtitle">Advanced flow matching architecture combining diffusion models with normalizing flows for superior image generation</p>
        </div>

        <!-- Architecture Display -->
        <div class="framework-main-showcase">
            <div class="framework-image-container">
                <div class="framework-image-wrapper">
                    <img src="/LucidFlux/architecture/arch.png" alt="LucidFlux Architecture" class="framework-image">
                    <div class="framework-glow"></div>
                </div>
            </div>
        </div>

        <!-- Key Features -->
        <div class="workflow-stages">
            <div class="stage-card stage-1" data-stage="1">
                <div class="stage-header">
                    <div class="stage-icon">🌊</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Flow Matching</h3>
                    </div>
                </div>
                <p class="stage-description">Advanced flow matching techniques combining diffusion models and normalizing flows for stable training and better mode coverage.</p>
            </div>
            
            <div class="stage-card stage-2" data-stage="2">
                <div class="stage-header">
                    <div class="stage-icon">🎨</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Enhanced U-Net</h3>
                    </div>
                </div>
                <p class="stage-description">Enhanced U-Net backbone with attention mechanisms and multi-modal conditioning for superior generation quality.</p>
            </div>
            
            <div class="stage-card stage-3" data-stage="3">
                <div class="stage-header">
                    <div class="stage-icon">⚡</div>
                    <div class="stage-info">
                        <h3 class="stage-title">High Resolution</h3>
                    </div>
                </div>
                <p class="stage-description">Progressive resolution scaling enabling high-resolution outputs up to 2048×2048 with computational efficiency.</p>
            </div>
            
            <div class="stage-card stage-4" data-stage="4">
                <div class="stage-header">
                    <div class="stage-icon">🔧</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Advanced Training</h3>
                    </div>
                </div>
                <p class="stage-description">Advanced loss functions and optimization strategies ensuring stable training and faster convergence.</p>
            </div>
        </div>
    </div>
</div>

<!-- 新增：Technical Overview Section - 全栏设计 -->
<div class="technical-overview-section" id="overview">
    <div class="overview-container">
        <div class="overview-header">
            <h2 class="overview-title">Technical Framework</h2>
            <div class="overview-emoji">⚡</div>
            <p class="overview-subtitle">Advanced flow matching architecture for high-resolution image generation through optimized diffusion models</p>
        </div>
        
        <!-- 主要框架图展示 -->
        <div class="framework-main-showcase">
            <div class="framework-image-container">
                <div class="framework-image-wrapper">
                    <img src="/LucidFlux/architecture/arch.png" alt="LucidFlux Technical Framework" class="framework-image">
                    <div class="framework-glow"></div>
                    
                    <!-- 交互式热点 -->
                    <div class="framework-hotspots">
                        <div class="hotspot hotspot-1" data-stage="1">
                            <div class="hotspot-pulse"></div>
                            <div class="hotspot-number">1</div>
                        </div>
                        <div class="hotspot hotspot-2" data-stage="2">
                            <div class="hotspot-pulse"></div>
                            <div class="hotspot-number">2</div>
                        </div>
                        <div class="hotspot hotspot-3" data-stage="3">
                            <div class="hotspot-pulse"></div>
                            <div class="hotspot-number">3</div>
                        </div>
                        <div class="hotspot hotspot-4" data-stage="4">
                            <div class="hotspot-pulse"></div>
                            <div class="hotspot-number">4</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- 四个阶段的详细展示 -->
        <div class="workflow-stages">
            <div class="stage-card stage-1" data-stage="1">
                <div class="stage-header">
                    <div class="stage-icon">🌊</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Flow Matching Architecture</h3>
                    </div>
                </div>
                <p class="stage-description">Implements advanced flow matching techniques that combine the best of diffusion models and normalizing flows, providing stable training dynamics and better mode coverage for high-quality image generation.</p>
            </div>
            
            <!-- 第一个箭头 -->
            <div class="stage-arrow stage-arrow-1">
                <div class="arrow-icon">➤</div>
            </div>
            
            <div class="stage-card stage-2" data-stage="2">
                <div class="stage-header">
                    <div class="stage-icon">🎨</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Enhanced U-Net Backbone</h3>
                    </div>
                </div>
                <p class="stage-description">Utilizes an enhanced U-Net architecture with advanced attention mechanisms and multi-modal conditioning support for superior image generation quality.</p>
            </div>
            
            <!-- 第二个箭头 -->
            <div class="stage-arrow stage-arrow-2">
                <div class="arrow-icon">➤</div>
            </div>
            
            <div class="stage-card stage-3" data-stage="3">
                <div class="stage-header">
                    <div class="stage-icon">⚡</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Progressive Resolution Scaling</h3>
                    </div>
                </div>
                <p class="stage-description">Employs progressive resolution scaling during training to achieve high-resolution outputs up to 2048×2048 while maintaining computational efficiency.</p>
            </div>
            
            <!-- 第三个箭头 -->
            <div class="stage-arrow stage-arrow-3">
                <div class="arrow-icon">➤</div>
            </div>
            
            <div class="stage-card stage-4" data-stage="4">
                <div class="stage-header">
                    <div class="stage-icon">🔧</div>
                    <div class="stage-info">
                        <h3 class="stage-title">Advanced Optimization</h3>
                    </div>
                </div>
                <p class="stage-description">Features advanced loss functions, regularization techniques, and optimization strategies to ensure stable training and superior sample quality with faster convergence.</p>
            </div>
        </div>
    </div>
</div>

<!-- Results Showcase Section -->
<div class="results-showcase-section" id="results-showcase">
    <div class="results-container">
        <div class="results-header">
            <h2 class="results-title">Generation Results</h2>
            <div class="results-emoji">🎨</div>
            <p class="results-subtitle">High-quality image generation showcasing LucidFlux capabilities</p>
        </div>
        
        <!-- Results Grid -->
        <div class="results-grid">
            <div class="result-item">
                <div class="result-image-frame">
                    <img src="/LucidFlux/gallery/concatenated_006.jpg" alt="LucidFlux Result 1" class="result-image">
                    <div class="result-glow"></div>
                </div>
                <div class="result-caption">Natural Landscape</div>
            </div>
            
            <div class="result-item">
                <div class="result-image-frame">
                    <img src="/LucidFlux/gallery/concatenated_019.jpg" alt="LucidFlux Result 2" class="result-image">
                    <div class="result-glow"></div>
                </div>
                <div class="result-caption">Artistic Portrait</div>
            </div>
            
            <div class="result-item">
                <div class="result-image-frame">
                    <img src="/LucidFlux/gallery/concatenated_040.jpg" alt="LucidFlux Result 3" class="result-image">
                    <div class="result-glow"></div>
                </div>
                <div class="result-caption">Abstract Art</div>
            </div>
            
            <div class="result-item">
                <div class="result-image-frame">
                    <img src="/LucidFlux/gallery/concatenated_082.jpg" alt="LucidFlux Result 4" class="result-image">
                    <div class="result-glow"></div>
                </div>
                <div class="result-caption">Architecture</div>
            </div>
        </div>
    </div>
</div>

<!-- Performance Metrics Section -->
<div class="performance-section" id="performance">
    <div class="performance-container">
        <div class="performance-header">
            <h2 class="performance-title">Performance Metrics</h2>
            <div class="performance-emoji">📊</div>
            <p class="performance-subtitle">Quantitative evaluation of LucidFlux generation quality</p>
        </div>
        
        <!-- Metrics Grid -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">12.3</div>
                <div class="metric-label">FID Score ↓</div>
                <div class="metric-desc">34% improvement</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">⭐</div>
                <div class="metric-value">8.2</div>
                <div class="metric-label">Inception Score ↑</div>
                <div class="metric-desc">19% improvement</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">🔗</div>
                <div class="metric-value">0.85</div>
                <div class="metric-label">CLIP Score ↑</div>
                <div class="metric-desc">9% improvement</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-icon">⚡</div>
                <div class="metric-value">2.1s</div>
                <div class="metric-label">Generation Time</div>
                <div class="metric-desc">2048×2048 resolution</div>
            </div>
        </div>
    </div>
</div>

<!-- Applications Section -->
<div class="applications-section" id="applications">
    <div class="applications-container">
        <div class="applications-header">
            <h2 class="applications-title">Applications</h2>
            <div class="applications-emoji">🚀</div>
            <p class="applications-subtitle">Versatile applications across different domains</p>
        </div>
        
        <div class="applications-grid">
            <div class="application-card">
                <div class="application-icon">🎨</div>
                <h3 class="application-title">Digital Art</h3>
                <p class="application-desc">Create stunning digital artwork with advanced style transfer and artistic effects</p>
            </div>
            
            <div class="application-card">
                <div class="application-icon">📸</div>
                <h3 class="application-title">Photo Enhancement</h3>
                <p class="application-desc">Enhance and restore images with superior quality and detail preservation</p>
            </div>
            
            <div class="application-card">
                <div class="application-icon">🎬</div>
                <h3 class="application-title">Content Creation</h3>
                <p class="application-desc">Generate high-quality content for media, marketing, and entertainment</p>
            </div>
            
            <div class="application-card">
                <div class="application-icon">🔬</div>
                <h3 class="application-title">Research</h3>
                <p class="application-desc">Advanced research in generative AI and computer vision applications</p>
            </div>
        </div>
    </div>
</div>

<!-- Gallery Section -->
<div class="gallery-section" id="gallery">
    <div class="gallery-container">
        <div class="gallery-header">
            <h2 class="gallery-title">Generation Gallery</h2>
            <div class="gallery-emoji">🎭</div>
            <p class="gallery-subtitle">Explore diverse high-quality images generated by LucidFlux</p>
        </div>
        
        <div class="gallery-grid">
            <!-- Gallery items with LucidFlux images -->
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_006.jpg" alt="LucidFlux Generated Landscape">
                <div class="gallery-caption">Natural Landscape</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_013.jpg" alt="LucidFlux Generated Portrait">
                <div class="gallery-caption">Artistic Portrait</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_019.jpg" alt="LucidFlux Generated Abstract">
                <div class="gallery-caption">Abstract Art</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_040.jpg" alt="LucidFlux Generated Scene">
                <div class="gallery-caption">Scene Composition</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_041.jpg" alt="LucidFlux Generated Architecture">
                <div class="gallery-caption">Architecture</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_079.jpg" alt="LucidFlux Generated Nature">
                <div class="gallery-caption">Nature Scene</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_082.jpg" alt="LucidFlux Generated Urban">
                <div class="gallery-caption">Urban Landscape</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_111.jpg" alt="LucidFlux Generated Concept">
                <div class="gallery-caption">Concept Art</div>
            </div>
            <div class="gallery-item">
                <img src="/LucidFlux/gallery/concatenated_123.jpg" alt="LucidFlux Generated Creative">
                <div class="gallery-caption">Creative Design</div>
            </div>
        </div>
    </div>
</div>
