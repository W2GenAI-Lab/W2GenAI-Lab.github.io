// LucidFlux HTML Script - Based on Turbo2K Functionality
document.addEventListener('DOMContentLoaded', function() {
    
    // Navigation functionality
    const navLinks = document.querySelectorAll('.nav-link');
    const sections = document.querySelectorAll('section');
    
    // Active state management
    function setActiveLink() {
        const scrollPosition = window.scrollY + 100;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }
    
    // Smooth scrolling
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            const targetSection = document.querySelector(targetId);
            
            if (targetSection) {
                const targetPosition = targetSection.offsetTop - 80;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Set active link on scroll
    window.addEventListener('scroll', setActiveLink);
    
    // Set active link on page load
    setActiveLink();
    
    // Mobile menu toggle
    const sidebar = document.querySelector('.sidebar');
    const mobileMenuToggle = document.createElement('button');
    mobileMenuToggle.className = 'mobile-menu-toggle';
    mobileMenuToggle.innerHTML = '☰';
    mobileMenuToggle.style.display = 'none';
    document.body.appendChild(mobileMenuToggle);
    
    mobileMenuToggle.addEventListener('click', function() {
        sidebar.classList.toggle('active');
    });
    
    // Check if we're on mobile
    function checkMobile() {
        if (window.innerWidth <= 768) {
            mobileMenuToggle.style.display = 'block';
            sidebar.style.transform = 'translateX(-100%)';
        } else {
            mobileMenuToggle.style.display = 'none';
            sidebar.style.transform = 'translateX(0)';
            sidebar.classList.remove('active');
        }
    }
    
    // Check on load and resize
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    // Gallery Modal functionality
    const modal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImage');
    const modalCaption = document.getElementById('modalCaption');
    const closeBtn = document.querySelector('.close');
    const galleryItems = document.querySelectorAll('.gallery-item');
    
    // Open modal when clicking on gallery items
    galleryItems.forEach(item => {
        item.addEventListener('click', function() {
            const img = this.querySelector('img');
            const caption = this.querySelector('.gallery-caption');
            
            modal.style.display = 'block';
            modalImg.src = img.src;
            modalImg.alt = img.alt;
            modalCaption.textContent = caption.textContent;
            
            // Disable body scroll
            document.body.style.overflow = 'hidden';
        });
    });
    
    // Close modal
    function closeModal() {
        modal.style.display = 'none';
        document.body.style.overflow = 'auto';
    }
    
    closeBtn.addEventListener('click', closeModal);
    
    // Close modal when clicking outside the image
    modal.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // Close modal with Escape key
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });
    
    // Add some visual feedback for gallery items
    galleryItems.forEach(item => {
        item.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        item.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
    
    // Smooth reveal animation for sections
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver(function(entries) {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe all sections
    sections.forEach(section => {
        section.style.opacity = '0';
        section.style.transform = 'translateY(20px)';
        section.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(section);
    });
    
    // Add parallax effect to architecture image
    const architectureImages = document.querySelectorAll('.architecture-showcase img');
    
    window.addEventListener('scroll', function() {
        const scrolled = window.pageYOffset;
        
        architectureImages.forEach(img => {
            const speed = 0.5;
            const yPos = -(scrolled * speed);
            img.style.transform = `translateY(${yPos}px)`;
        });
    });
    
    // Add loading animation for images
    const images = document.querySelectorAll('img[loading="lazy"]');
    
    images.forEach(img => {
        img.addEventListener('load', function() {
            this.style.opacity = '1';
        });
        
        img.addEventListener('error', function() {
            this.style.opacity = '0.5';
        });
    });
    
    // Initialize tooltips for metric descriptions
    const metricDescs = document.querySelectorAll('.metric-desc');
    
    metricDescs.forEach(desc => {
        desc.title = desc.textContent;
    });
    
    // Add keyboard navigation for gallery
    let currentImageIndex = 0;
    
    document.addEventListener('keydown', function(e) {
        if (modal.style.display === 'block') {
            if (e.key === 'ArrowLeft') {
                currentImageIndex = (currentImageIndex - 1 + galleryItems.length) % galleryItems.length;
                showImage(currentImageIndex);
            } else if (e.key === 'ArrowRight') {
                currentImageIndex = (currentImageIndex + 1) % galleryItems.length;
                showImage(currentImageIndex);
            }
        }
    });
    
    function showImage(index) {
        const item = galleryItems[index];
        const img = item.querySelector('img');
        const caption = item.querySelector('.gallery-caption');
        
        modalImg.src = img.src;
        modalImg.alt = img.alt;
        modalCaption.textContent = caption.textContent;
        
        // Add fade effect
        modalImg.style.opacity = '0';
        setTimeout(() => {
            modalImg.style.opacity = '1';
        }, 100);
    }
    
    // Update current image index when clicking on gallery items
    galleryItems.forEach((item, index) => {
        item.addEventListener('click', function() {
            currentImageIndex = index;
        });
    });
    
    // Add smooth scroll behavior for modern browsers
    if ('scrollBehavior' in document.documentElement.style) {
        // Modern browsers support smooth scrolling natively
        console.log('Smooth scrolling supported');
    } else {
        // Fallback for older browsers
        console.log('Smooth scrolling not supported');
    }
    
    // Performance optimization: throttle scroll events
    function throttle(func, limit) {
        let inThrottle;
        return function() {
            const args = arguments;
            const context = this;
            if (!inThrottle) {
                func.apply(context, args);
                inThrottle = true;
                setTimeout(() => inThrottle = false, limit);
            }
        }
    }
    
    // Apply throttling to scroll events
    window.addEventListener('scroll', throttle(setActiveLink, 100));
    
    // Add print styles
    window.addEventListener('beforeprint', function() {
        document.body.classList.add('print-mode');
    });
    
    window.addEventListener('afterprint', function() {
        document.body.classList.remove('print-mode');
    });
    
    // Console welcome message
    console.log('%cLucidFlux 🌊', 'color: #3498db; font-size: 20px; font-weight: bold;');
    console.log('%cAdvanced Image Generation with Flow Matching', 'color: #2c3e50; font-size: 14px;');
    console.log('%cGitHub: https://github.com/Ephemeral182/PosterCraft', 'color: #27ae60; font-size: 12px;');
    
});