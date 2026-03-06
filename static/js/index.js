// BibTeX copy
function copyBibtex() {
  const text = document.querySelector('.bibtex-box code').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = document.querySelector('.copy-btn');
    btn.innerHTML = '<i class="fas fa-check"></i> Copied!';
    setTimeout(() => { btn.innerHTML = '<i class="fas fa-copy"></i> Copy'; }, 2000);
  });
}

// Qualitative carousel
const carouselState = {};

function initCarousels() {
  document.querySelectorAll('.qual-carousel').forEach(carousel => {
    const backbone = carousel.id.replace('carousel-', '');
    const slides = carousel.querySelectorAll('.carousel-slide');
    carouselState[backbone] = { current: 0, total: slides.length };

    const dotsContainer = document.getElementById('dots-' + backbone);
    slides.forEach((_, i) => {
      const dot = document.createElement('button');
      dot.className = 'carousel-dot' + (i === 0 ? ' active' : '');
      dot.onclick = () => goToSlide(backbone, i);
      dotsContainer.appendChild(dot);
    });
  });
}

function goToSlide(backbone, index) {
  const carousel = document.getElementById('carousel-' + backbone);
  const slides = carousel.querySelectorAll('.carousel-slide');
  const dots = document.querySelectorAll('#dots-' + backbone + ' .carousel-dot');

  slides[carouselState[backbone].current].classList.remove('active');
  dots[carouselState[backbone].current].classList.remove('active');

  carouselState[backbone].current = (index + carouselState[backbone].total) % carouselState[backbone].total;

  slides[carouselState[backbone].current].classList.add('active');
  dots[carouselState[backbone].current].classList.add('active');
}

function carouselStep(backbone, direction) {
  goToSlide(backbone, carouselState[backbone].current + direction);
}

document.addEventListener('DOMContentLoaded', () => {
  initCarousels();
});
