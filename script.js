
      // Mobile menu
      (function () {
        const btn = document.getElementById('menuBtn');
        const drawer = document.getElementById('mobileNav');
        if (!btn || !drawer) return;
        const setOpen = (open) => {
          btn.setAttribute('aria-expanded', String(open));
          drawer.style.display = open ? 'block' : 'none';
        };
        btn.addEventListener('click', () => {
          const open = btn.getAttribute('aria-expanded') !== 'true';
          setOpen(open);
        });
        drawer.addEventListener('click', (e) => {
          const a = e.target.closest('a[href^="#"]');
          if (a) setOpen(false);
        });
        let last = window.innerWidth;
        window.addEventListener('resize', () => {
          const now = window.innerWidth;
          if (now >= 960 && last < 960) setOpen(false);
          last = now;
        });
      })();
function openPopup() {
    window.open(
      "run.html",       
      "popupWindow",     
      "width=1200,height=1000,top=100,left=100,resizable=yes,scrollbars=yes"
    );
  }
      // Smooth scroll with reduced motion respect
      (function () {
        const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        document.addEventListener('click', (e) => {
          const a = e.target.closest('a[href^="#"]');
          if (!a) return;
          const id = a.getAttribute('href').slice(1);
          if (!id) return;
          const el = document.getElementById(id);
          if (!el) return;
          e.preventDefault();
          el.scrollIntoView({ behavior: prefersReduced ? 'auto' : 'smooth', block: 'start' });
          // Manage focus for accessibility
          el.setAttribute('tabindex', '-1');
          el.focus({ preventScroll: true });
        });
      })();

      // Reveal on scroll (fade-up)
      (function () {
        const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (prefersReduced) return; // already visible via CSS fallback
        const els = document.querySelectorAll('.reveal');
        const io = new IntersectionObserver((entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              entry.target.classList.add('in-view');
              io.unobserve(entry.target);
            }
          });
        }, { threshold: 0.2 });
        els.forEach((el) => io.observe(el));
      })();

      // FAQ accordion (animated height)
      (function () {
        const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
        function toggle(panel, expanded) {
          const start = panel.scrollHeight;
          if (expanded) {
            panel.style.maxHeight = '0px';
          } else {
            panel.style.maxHeight = panel.scrollHeight + 'px';
          }
          // If content changes after open, adjust height
          if (!prefersReduced && !expanded) {
            requestAnimationFrame(() => {
              panel.style.maxHeight = panel.scrollHeight + 'px';
            });
          }
        }

        document.querySelectorAll('.faq-item').forEach((item, idx) => {
          const btn = item.querySelector('.faq-btn');
          const panel = item.querySelector('.faq-panel');
          if (!btn || !panel) return;
          btn.addEventListener('click', () => {
            const expanded = btn.getAttribute('aria-expanded') === 'true';
            btn.setAttribute('aria-expanded', String(!expanded));
            toggle(panel, expanded);
          });
          // Ensure collapsed on load
          btn.setAttribute('aria-expanded', 'false');
          panel.style.maxHeight = '0px';
        });
      })();

      // Current year
      document.getElementById('year').textContent = new Date().getFullYear();
