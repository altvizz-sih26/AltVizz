import { useEffect, useRef } from 'react';

export default function Stardust() {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas.getContext('2d');

    const reduced = window.matchMedia(
      '(prefers-reduced-motion: reduce)'
    ).matches;

    const phoneQuery = window.matchMedia('(max-width: 430px)');

    let width = 0;
    let height = 0;
    let frame = 0;
    let stars = [];
    let isPhone = phoneQuery.matches;
    let lastTimestamp = 0;

    const makeStars = () => {
      /*
       * Match the reference hero:
       * 420 total stars on desktop.
       * Keep the mobile count high enough so the phone
       * does not look empty.
       */
      const count = isPhone ? 360 : 420;

      stars = Array.from({ length: count }, () => {
        /*
         * Reference:
         * const isOrb = Math.random() > 0.95;
         */
        const isOrb = Math.random() > 0.95;

        /*
         * Reference:
         * Normal: 0.8 - 2.3px
         * Orb:    4 - 7px
         */
        const size = isOrb
          ? 4 + Math.random() * 3
          : 0.8 + Math.random() * 1.5;

        /*
         * Reference falling durations:
         * Normal stars: 20 - 70 seconds
         * Orbs:         60 - 140 seconds
         *
         * We use the slower portion of these ranges so
         * the motion remains soft and atmospheric.
         */
        const fallDuration = isOrb
          ? 100 + Math.random() * 40
          : 45 + Math.random() * 25;

        /*
         * Reference twinkle:
         * Normal: 3 - 8 seconds
         * Orbs:   6 - 16 seconds
         */
        const twinkleDuration = isOrb
          ? 6 + Math.random() * 10
          : 3 + Math.random() * 5;

        return {
          x: Math.random() * width,

          /*
           * Start throughout the screen instead of making
           * all stars enter from the top together.
           */
          y: Math.random() * height,

          size,
          isOrb,

          fallDuration,
          twinkleDuration,

          /*
           * Random phase prevents synchronized animation.
           */
          fallPhase: Math.random(),
          twinklePhase: Math.random(),

          /*
           * Small horizontal movement, similar to the
           * atmospheric feeling of the reference.
           */
          drift: (Math.random() - 0.5) * 0.12,

          /*
           * Reference stars begin with different opacity.
           */
          alpha: 0.45 + Math.random() * 0.55,

          /*
           * Slight individual variation.
           */
          twinkleOffset: Math.random() * Math.PI * 2,
        };
      });
    };

    const resize = () => {
      isPhone = phoneQuery.matches;

      width = window.innerWidth;
      height = window.innerHeight;

      const dpr = Math.min(
        window.devicePixelRatio || 1,
        isPhone ? 2 : 1.75
      );

      canvas.width = Math.max(1, Math.round(width * dpr));
      canvas.height = Math.max(1, Math.round(height * dpr));

      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      makeStars();
    };

    const draw = (time) => {
      if (!lastTimestamp) {
        lastTimestamp = time;
      }

      const deltaSeconds = Math.min(
        (time - lastTimestamp) / 1000,
        0.05
      );

      lastTimestamp = time;

      ctx.clearRect(0, 0, width, height);

      if (!reduced) {
        for (const star of stars) {
          /*
           * ==============================================
           * FALLING
           * ==============================================
           *
           * Each star completes a very slow downward
           * journey over its own duration.
           */
          const fallProgress =
            ((time / 1000) / star.fallDuration + star.fallPhase) % 1;

          /*
           * Start slightly above the viewport and finish
           * slightly below it.
           */
          const y =
            -height * 0.1 +
            fallProgress * (height * 1.2);

          /*
           * ==============================================
           * TWINKLING
           * ==============================================
           *
           * This reproduces the reference:
           *
           * 0%   -> opacity 0, scale 0.5
           * 50%  -> opacity 1, scale 1.2
           * 100% -> opacity 0, scale 0.5
           *
           * Crucially, this is calculated independently
           * from the falling movement.
           */
          const twinkleProgress =
            ((time / 1000) / star.twinkleDuration +
              star.twinklePhase) %
            1;

          const twinkleWave =
            (Math.sin(
              twinkleProgress * Math.PI * 2 +
                star.twinkleOffset
            ) +
              1) /
            2;

          /*
           * Make the fade soft rather than abruptly blinking.
           */
          const smoothTwinkle =
            twinkleWave * twinkleWave;

          /*
           * Reference starts at 0 opacity and peaks at 1.
           */
          const opacity =
            star.alpha *
            (0.12 + smoothTwinkle * 0.88);

          /*
           * Reference scales:
           * 0.5 -> 1.2 -> 0.5
           */
          const scale =
            0.5 + smoothTwinkle * 0.7;

          /*
           * Very subtle horizontal atmospheric drift.
           */
          const driftX =
            Math.sin(
              time * 0.00012 +
                star.twinkleOffset
            ) *
            (star.isOrb ? 1.8 : 1.2);

          const x =
            star.x +
            driftX +
            star.drift *
              ((time - star.twinklePhase * 1000) % 100000) /
              1000;

          /*
           * ==============================================
           * DRAW STAR
           * ==============================================
           */

          ctx.save();

          if (star.isOrb) {
            /*
             * Larger glowing stars from the reference.
             */
            const radius = star.size * scale;

            ctx.shadowBlur = 14;
            ctx.shadowColor =
              'rgba(255,255,255,0.8)';

            ctx.beginPath();

            ctx.fillStyle = `rgba(255,255,255,${opacity * 0.8})`;

            ctx.arc(
              x,
              y,
              radius,
              0,
              Math.PI * 2
            );

            ctx.fill();

            /*
             * Soft warm outer glow.
             */
            ctx.shadowBlur = 0;

            ctx.beginPath();

            ctx.fillStyle = `rgba(212,188,150,${opacity * 0.32})`;

            ctx.arc(
              x,
              y,
              radius * 2.8,
              0,
              Math.PI * 2
            );

            ctx.fill();
          } else {
            /*
             * Tiny normal stars.
             */
            const radius = Math.max(
              0.4,
              star.size * scale
            );

            ctx.shadowBlur = 0;

            ctx.beginPath();

            ctx.fillStyle = `rgba(255,255,255,${opacity})`;

            ctx.arc(
              x,
              y,
              radius,
              0,
              Math.PI * 2
            );

            ctx.fill();
          }

          ctx.restore();
        }
      }

      ctx.shadowBlur = 0;

      frame = requestAnimationFrame(draw);
    };

    resize();

    window.addEventListener('resize', resize);

    frame = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      className="stardust"
      ref={ref}
      aria-hidden="true"
    />
  );
}