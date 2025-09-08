---
layout: default
permalink: /LucidFlux/
title: LucidFlux - Advanced Image Generation with Flow Matching
extra_css:
  - /assets/css/lucidflux.css
extra_js:
  - /assets/js/lucidflux.js
---

<div class="project-container">
    <!-- Sidebar Navigation -->
    <div class="sidebar">
        <div class="sidebar-header">
            <h1 class="project-title">LucidFlux</h1>
            <p class="project-subtitle">Advanced Image Generation</p>
        </div>
        
        <nav class="sidebar-nav">
            <ul class="nav-list">
                <li class="nav-item">
                    <a href="#paper" class="nav-link">Paper Information</a>
                </li>
                <li class="nav-item">
                    <a href="#overview" class="nav-link">Overview</a>
                </li>
                <li class="nav-item">
                    <a href="#architecture" class="nav-link">Technical Framework</a>
                </li>
                <li class="nav-item">
                    <a href="#results" class="nav-link">Results Gallery</a>
                </li>
                <li class="nav-item">
                    <a href="#performance" class="nav-link">Performance</a>
                </li>
                <li class="nav-item">
                    <a href="#applications" class="nav-link">Applications</a>
                </li>
                <li class="nav-item">
                    <a href="#contact" class="nav-link">Contact</a>
                </li>
            </ul>
        </nav>
        
        <div class="sidebar-links">
            <a href="https://arxiv.org/abs/2506.10741" class="sidebar-link" target="_blank">
                <span class="link-icon">📄</span>
                Paper
            </a>
            <a href="https://github.com/Ephemeral182/PosterCraft" class="sidebar-link" target="_blank">
                <span class="link-icon">💻</span>
                Code
            </a>
        </div>
    </div>
    
    <!-- Main Content -->
    <main class="main-content">
        <!-- Paper Information Section -->
        <section id="paper" class="content-section">
            <h2 class="section-title">Paper Information</h2>
            <div class="paper-details">
                <h3 class="paper-title">Universal Image Restoration with a Large-Scale Diffusion Transformer</h3>
                
                <div class="authors">
                    <p>
                        <span class="author"><a href="https://ephemeral182.github.io/">Sixiang Chen</a><sup>1,2,*</sup></span>,
                        <span class="author"><a href="https://openreview.net/profile?id=~Jianyu_Lai1">Jianyu Lai</a><sup>1,*</sup></span>,
                        <span class="author"><a href="https://scholar.google.com/citations?user=sj4FqEgAAAAJ&hl=zh-CN">Jialin Gao</a><sup>2,*</sup></span>,
                        <span class="author"><a href="https://owen718.github.io/">Tian Ye</a><sup>1</sup></span>,
                        <span class="author"><a href="https://haoyuchen.com/">Haoyu Chen</a><sup>1</sup></span>
                    </p>
                    <p class="affiliations">
                        <sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou),
                        <sup>2</sup>Meituan
                    </p>
                </div>
                
                <div class="paper-links">
                    <a href="https://arxiv.org/abs/2506.10741" class="paper-link" target="_blank">arXiv</a>
                    <a href="https://github.com/Ephemeral182/PosterCraft" class="paper-link" target="_blank">GitHub</a>
                    <a href="https://huggingface.co/PosterCraft/PosterCraft-v1_RL" class="paper-link" target="_blank">HuggingFace</a>
                    <a href="https://huggingface.co/spaces/Ephemeral182/PosterCraft" class="paper-link" target="_blank">Demo</a>
                </div>
            </div>
        </section>
        
        <!-- Overview Section -->
        <section id="overview" class="content-section">
            <h2 class="section-title">Overview</h2>
            <div class="overview-content">
                <p>
                    LucidFlux is an advanced image generation model that leverages flow matching technology 
                    combined with diffusion models to generate high-resolution, high-quality images. 
                    Our approach focuses on stable training dynamics, efficient memory usage, and superior image quality.
                </p>
                
                <div class="key-features">
                    <div class="feature-item">
                        <h4>🌊 Flow Matching</h4>
                        <p>Advanced flow matching techniques combining diffusion models and normalizing flows for stable training</p>
                    </div>
                    <div class="feature-item">
                        <h4>🎨 Enhanced U-Net</h4>
                        <p>Enhanced U-Net backbone with attention mechanisms for superior generation quality</p>
                    </div>
                    <div class="feature-item">
                        <h4>⚡ High Resolution</h4>
                        <p>Progressive resolution scaling enabling outputs up to 2048×2048 with computational efficiency</p>
                    </div>
                    <div class="feature-item">
                        <h4>🔧 Advanced Training</h4>
                        <p>Advanced loss functions and optimization strategies ensuring stable training and faster convergence</p>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Technical Framework Section -->
        <section id="architecture" class="content-section">
            <h2 class="section-title">Technical Framework</h2>
            <div class="framework-content">
                <div class="architecture-image">
                    <img src="/LucidFlux/architecture/arch.png" alt="LucidFlux Architecture">
                </div>
                
                <div class="workflow-description">
                    <h3>Architecture Overview</h3>
                    <p>
                        LucidFlux implements a sophisticated flow matching architecture that combines the best 
                        of diffusion models and normalizing flows. The framework features an enhanced U-Net backbone 
                        with advanced attention mechanisms and multi-modal conditioning support.
                    </p>
                    <p>
                        Our progressive resolution scaling approach enables high-resolution outputs while maintaining 
                        computational efficiency, making it suitable for both research and production applications.
                    </p>
                </div>
            </div>
        </section>
        
        <!-- Results Gallery Section -->
        <section id="results" class="content-section">
            <h2 class="section-title">Results Gallery</h2>
            <div class="results-grid">
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_006.jpg" alt="Generation Result 1">
                    <div class="result-caption">Natural Landscape</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_013.jpg" alt="Generation Result 2">
                    <div class="result-caption">Artistic Portrait</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_019.jpg" alt="Generation Result 3">
                    <div class="result-caption">Abstract Art</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_040.jpg" alt="Generation Result 4">
                    <div class="result-caption">Architecture</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_041.jpg" alt="Generation Result 5">
                    <div class="result-caption">Scene Composition</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_079.jpg" alt="Generation Result 6">
                    <div class="result-caption">Nature Scene</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_082.jpg" alt="Generation Result 7">
                    <div class="result-caption">Urban Landscape</div>
                </div>
                <div class="result-item">
                    <img src="/LucidFlux/gallery/concatenated_111.jpg" alt="Generation Result 8">
                    <div class="result-caption">Concept Art</div>
                </div>
            </div>
        </section>
        
        <!-- Performance Section -->
        <section id="performance" class="content-section">
            <h2 class="section-title">Performance Metrics</h2>
            <div class="performance-metrics">
                <div class="metrics-grid">
                    <div class="metric">
                        <div class="metric-value">12.3</div>
                        <div class="metric-label">FID Score ↓</div>
                        <div class="metric-desc">34% improvement</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">8.2</div>
                        <div class="metric-label">Inception Score ↑</div>
                        <div class="metric-desc">19% improvement</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">0.85</div>
                        <div class="metric-label">CLIP Score ↑</div>
                        <div class="metric-desc">9% improvement</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">2.1s</div>
                        <div class="metric-label">Generation Time</div>
                        <div class="metric-desc">2048×2048 resolution</div>
                    </div>
                </div>
            </div>
        </section>
        
        <!-- Applications Section -->
        <section id="applications" class="content-section">
            <h2 class="section-title">Applications</h2>
            <div class="applications-grid">
                <div class="application">
                    <h4>🎨 Digital Art</h4>
                    <p>Create stunning digital artwork with advanced style transfer and artistic effects</p>
                </div>
                <div class="application">
                    <h4>📸 Photo Enhancement</h4>
                    <p>Enhance and restore images with superior quality and detail preservation</p>
                </div>
                <div class="application">
                    <h4>🎬 Content Creation</h4>
                    <p>Generate high-quality content for media, marketing, and entertainment</p>
                </div>
                <div class="application">
                    <h4>🔬 Research</h4>
                    <p>Advanced research in generative AI and computer vision applications</p>
                </div>
            </div>
        </section>
        
        <!-- Contact Section -->
        <section id="contact" class="content-section">
            <h2 class="section-title">Contact</h2>
            <div class="contact-info">
                <p>
                    For questions about LucidFlux, please contact the authors or visit our 
                    <a href="https://github.com/Ephemeral182/PosterCraft" target="_blank">GitHub repository</a>.
                </p>
                <p>
                    <strong>Corresponding Author:</strong> Lei Zhu (<a href="mailto:leizhu@ust.hk">leizhu@ust.hk</a>)
                </p>
            </div>
        </section>
    </main>
</div>
