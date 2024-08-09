document.addEventListener("DOMContentLoaded",function(){  
    
    //TODO: animate the hero title
    const heroTitle = document.querySelector('.hero');
    heroTitle.style.opacity = 0;
    heroTitle.style.transform = 'translateY(-50px)';

    setTimeout(() => {
        heroTitle.style.transition = 'opacity 1s ease-out, transform 1s ease-out';
        heroTitle.style.opacity = 1;
        heroTitle.style.transform = 'translateY(0)';
    },100);
});