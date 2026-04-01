import { useEffect, useRef } from 'react';
import { useUser, UserButton } from '@clerk/clerk-react';
import { useNavigate } from 'react-router-dom';
import { gsap } from 'gsap';
import {
  Shield, Camera, Activity, ShieldAlert,
  ChevronRight, BarChart3, Database,
  User, CheckCircle2, Zap, ArrowUpRight, Layers, Video
} from 'lucide-react';

/* ─────────────────────────────────────────────
   DESIGN SYSTEM — Forensic Terminal Luxury
   Tone: Luxury Minimal × Industrial Utilitarian
   DFII: 13/15
   Anchor: Asymmetric bento + dominant hero stat card
   Typography: Syne (display) + DM Sans (body)
───────────────────────────────────────────── */

export default function Dashboard() {
  /* ── auth & navigation (untouched) ── */
  const { user, isLoaded } = useUser();
  const navigate = useNavigate();

  /* ── GSAP refs (untouched) ── */
  const headerRef = useRef(null);
  const cardsRef = useRef([]);

  useEffect(() => {
    const tl = gsap.timeline({ defaults: { ease: 'power3.out' } });
    tl.fromTo(headerRef.current,
      { opacity: 0, y: -20 },
      { opacity: 1, y: 0, duration: 0.8 }
    )
    .fromTo(cardsRef.current,
      { opacity: 0, y: 20 },
      { opacity: 1, y: 0, duration: 0.8, stagger: 0.1 },
      '-=0.6'
    );
  }, []);

  const addToRefs = (el) => {
    if (el && !cardsRef.current.includes(el)) {
      cardsRef.current.push(el);
    }
  };

  /* ── Clerk loading guard (untouched) ── */
  if (!isLoaded) return null;

  return (
    <div
      style={{ fontFamily: "'DM Sans', sans-serif" }}
      className="h-screen w-screen bg-[#080808] text-white flex flex-col overflow-hidden"
    >
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

        /* ambient orbs */
        @keyframes drift-a {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(40px, -30px) scale(1.08); }
        }
        @keyframes drift-b {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(-35px, 25px) scale(1.06); }
        }
        .orb-a { animation: drift-a 14s ease-in-out infinite; }
        .orb-b { animation: drift-b 18s ease-in-out infinite; }

        /* live pulse badge */
        @keyframes pulse-ring {
          0%   { transform: scale(1);   opacity: 0.7; }
          100% { transform: scale(2.2); opacity: 0;   }
        }
        .pulse-ring { animation: pulse-ring 1.8s ease-out infinite; }

        /* shimmer CTA */
        @keyframes shimmer-g {
          0%   { background-position: -200% 0; }
          100% { background-position:  200% 0; }
        }
        .btn-shimmer {
          background: linear-gradient(90deg,#059669 0%,#34d399 40%,#059669 100%);
          background-size: 200% 100%;
          animation: shimmer-g 2.8s linear infinite;
        }

        /* stat counter fade-up */
        @keyframes count-up {
          from { opacity: 0; transform: translateY(10px); }
          to   { opacity: 1; transform: translateY(0); }
        }
        .stat-val { animation: count-up 0.6s ease forwards; }

        /* noise texture */
        .noise::after {
          content: '';
          position: absolute;
          inset: 0;
          background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
          background-size: 180px;
          pointer-events: none;
          border-radius: inherit;
          mix-blend-mode: overlay;
        }

        ::-webkit-scrollbar       { width: 3px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(52,211,153,.12); border-radius: 10px; }
      `}</style>

      {/* Ambient background glows */}
      <div style={{ position: 'fixed', inset: 0, overflow: 'hidden', pointerEvents: 'none', zIndex: 0 }}>
        <div className="orb-a" style={{
          position: 'absolute', top: '-10%', right: '5%',
          width: 600, height: 600, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(52,211,153,0.06) 0%, transparent 70%)',
        }} />
        <div className="orb-b" style={{
          position: 'absolute', bottom: '-15%', left: '-5%',
          width: 700, height: 700, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(34,211,238,0.04) 0%, transparent 70%)',
        }} />
      </div>

      {/* ══════════════════════════════════════
          HEADER
      ══════════════════════════════════════ */}
      <header
        ref={headerRef}
        style={{
          flexShrink: 0,
          height: 60,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '0 28px',
          background: 'rgba(8,8,8,0.85)',
          backdropFilter: 'blur(20px)',
          borderBottom: '1px solid rgba(255,255,255,0.05)',
          zIndex: 50,
          position: 'relative',
        }}
      >
        {/* Logo — click to go home */}
        <button
          onClick={() => navigate('/')}
          style={{
            display: 'flex', alignItems: 'center', gap: 10,
            background: 'none', border: 'none', cursor: 'pointer',
            padding: '4px 8px', borderRadius: 10,
            transition: 'background 0.15s',
          }}
          onMouseEnter={e => e.currentTarget.style.background = 'rgba(255,255,255,0.04)'}
          onMouseLeave={e => e.currentTarget.style.background = 'none'}
        >
          <div style={{
            width: 32, height: 32, borderRadius: 10,
            background: 'linear-gradient(135deg,#34d399,#22d3ee)',
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            position: 'relative',
          }}>
            <Shield size={14} color="#080808" strokeWidth={2.5} />
            <span style={{
              position: 'absolute', top: -2, right: -2,
              width: 8, height: 8, borderRadius: '50%',
              background: '#34d399', border: '2px solid #080808',
            }} />
          </div>
          <div>
            <p style={{
              fontFamily: 'Syne, sans-serif', fontWeight: 700,
              fontSize: 13, color: '#fff', lineHeight: 1, letterSpacing: '-0.01em',
            }}>
              DeepSheild<span style={{ color: '#34d399' }}>.ai</span>
            </p>
            <p style={{ fontSize: 10, color: '#3f3f46', marginTop: 2, lineHeight: 1 }}>
              Command Center
            </p>
          </div>
        </button>

        {/* Right side */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          {/* System status pill */}
          <div style={{
            display: 'flex', alignItems: 'center', gap: 7,
            padding: '5px 12px', borderRadius: 999,
            border: '1px solid rgba(52,211,153,0.2)',
            background: 'rgba(52,211,153,0.06)',
            fontSize: 11, color: '#34d399', fontWeight: 500,
          }}>
            <span style={{ position: 'relative', width: 8, height: 8, display: 'flex' }}>
              <span className="pulse-ring" style={{
                position: 'absolute', inset: 0,
                borderRadius: '50%', background: '#34d399',
              }} />
              <span style={{
                position: 'relative', width: 8, height: 8,
                borderRadius: '50%', background: '#34d399',
              }} />
            </span>
            System Online
          </div>
          <UserButton
            afterSignOutUrl="/"
            appearance={{
              elements: { userButtonAvatarBox: 'w-8 h-8 border border-emerald-500/30' }
            }}
          />
        </div>
      </header>

      {/* ══════════════════════════════════════
          MAIN SCROLL AREA
      ══════════════════════════════════════ */}
      <main style={{
        flex: 1, overflowY: 'auto', overflowX: 'hidden',
        position: 'relative', zIndex: 10,
      }}>
        <div style={{
          maxWidth: 1200, margin: '0 auto',
          padding: '36px 28px 48px',
          display: 'flex', flexDirection: 'column', gap: 28,
        }}>

          {/* ════ GREETING ROW ════ */}
          <div ref={addToRefs} style={{
            display: 'flex', alignItems: 'flex-end',
            justifyContent: 'space-between', flexWrap: 'wrap', gap: 20,
            paddingBottom: 24,
            borderBottom: '1px solid rgba(255,255,255,0.05)',
          }}>
            {/* Greeting text */}
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                <span style={{
                  padding: '3px 10px', borderRadius: 999,
                  background: 'rgba(52,211,153,0.08)',
                  border: '1px solid rgba(52,211,153,0.18)',
                  fontSize: 10, fontWeight: 600,
                  color: '#34d399', letterSpacing: '0.06em',
                  textTransform: 'uppercase',
                }}>
                  Access Granted
                </span>
                <span style={{ fontSize: 11, color: '#3f3f46', fontFamily: 'monospace' }}>
                  ID: {user?.id?.slice(5, 13).toUpperCase()}
                </span>
              </div>
              <h1 style={{
                fontFamily: 'Syne, sans-serif', fontWeight: 800,
                fontSize: 'clamp(24px, 4vw, 38px)',
                letterSpacing: '-0.04em', lineHeight: 1.1,
                color: '#fff', margin: 0,
              }}>
                Welcome back,{' '}
                <span style={{
                  background: 'linear-gradient(135deg,#34d399,#22d3ee)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                }}>
                  {user?.firstName || 'Agent'}
                </span>
                .
              </h1>
              <p style={{
                fontSize: 13, color: '#71717a', marginTop: 8,
                maxWidth: 440, lineHeight: 1.6,
              }}>
                Your workspace is primed. Monitor threats or launch a new deepfake detection session.
              </p>
            </div>

            {/* Launch Studio CTA */}
            <button
              onClick={() => navigate('/detector')}
              className="btn-shimmer"
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '12px 22px', borderRadius: 14,
                border: 'none', color: '#080808',
                fontSize: 13, fontWeight: 700,
                cursor: 'pointer', flexShrink: 0,
                fontFamily: 'DM Sans, sans-serif',
                boxShadow: '0 0 30px rgba(52,211,153,0.25)',
              }}
              onMouseEnter={e => e.currentTarget.style.transform = 'translateY(-2px)'}
              onMouseLeave={e => e.currentTarget.style.transform = 'translateY(0)'}
            >
              <Camera size={14} />
              Launch Studio
              <ArrowUpRight size={13} />
            </button>
          </div>

          {/* ════ BENTO GRID ════
              Asymmetric: [big hero] [2-col right]
              Row 2: [recent scans full width]
          ════════════════════════ */}
          <div ref={addToRefs} style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr 1fr',
            gridTemplateRows: 'auto',
            gap: 16,
          }}>

            {/* ── HERO STAT — Model Accuracy (spans 1 col, taller) ── */}
            <div className="noise" style={{
              position: 'relative', overflow: 'hidden',
              gridColumn: '1 / 2', gridRow: '1 / 2',
              padding: 28, borderRadius: 20,
              border: '1px solid rgba(52,211,153,0.12)',
              background: 'rgba(10,10,10,0.9)',
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              minHeight: 190,
            }}>
              {/* Glow behind number */}
              <div style={{
                position: 'absolute', right: -30, top: -30,
                width: 160, height: 160, borderRadius: '50%',
                background: 'radial-gradient(circle, rgba(52,211,153,0.12) 0%, transparent 70%)',
                pointerEvents: 'none',
              }} />
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 14 }}>
                  <ShieldAlert size={12} color="#34d399" />
                  <p style={{
                    fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
                    textTransform: 'uppercase', color: '#34d399',
                  }}>Model Accuracy</p>
                </div>
                <h3 className="stat-val" style={{
                  fontFamily: 'Syne, sans-serif', fontWeight: 800,
                  fontSize: 52, letterSpacing: '-0.05em', lineHeight: 1,
                  color: '#fff', margin: 0,
                }}>
                  96.64<span style={{ fontSize: 24, color: '#52525b' }}>%</span>
                </h3>
                <p style={{ fontSize: 12, color: '#52525b', marginTop: 8 }}>
                  Global threat baseline
                </p>
              </div>
              <div style={{
                display: 'flex', alignItems: 'center', gap: 6,
                padding: '6px 12px', borderRadius: 999, width: 'fit-content',
                border: '1px solid rgba(52,211,153,0.15)',
                background: 'rgba(52,211,153,0.06)',
              }}>
                <CheckCircle2 size={11} color="#34d399" />
                <span style={{ fontSize: 11, color: '#34d399' }}>Verified</span>
              </div>
            </div>

            {/* ── ENGINE STATUS ── */}
            <div className="noise" style={{
              position: 'relative', overflow: 'hidden',
              padding: 24, borderRadius: 20,
              border: '1px solid rgba(255,255,255,0.06)',
              background: 'rgba(10,10,10,0.9)',
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              minHeight: 190,
            }}>
              <div style={{ position: 'absolute', right: -20, top: -20, opacity: 0.08 }}>
                <Activity size={88} color="#34d399" />
              </div>
              <div>
                <p style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
                  textTransform: 'uppercase', color: '#52525b', marginBottom: 14,
                }}>Engine Status</p>
                <h3 className="stat-val" style={{
                  fontFamily: 'Syne, sans-serif', fontWeight: 800,
                  fontSize: 30, letterSpacing: '-0.03em', color: '#34d399', margin: 0,
                }}>Optimal</h3>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ position: 'relative', width: 8, height: 8, display: 'flex' }}>
                  <span className="pulse-ring" style={{
                    position: 'absolute', inset: 0,
                    borderRadius: '50%', background: '#34d399',
                  }} />
                  <span style={{
                    position: 'relative', width: 8, height: 8,
                    borderRadius: '50%', background: '#34d399',
                  }} />
                </span>
                <p style={{ fontSize: 12, color: '#71717a' }}>Latency &lt; 50ms</p>
              </div>
            </div>

            {/* ── SCANS PERFORMED ── */}
            <div className="noise" style={{
              position: 'relative', overflow: 'hidden',
              padding: 24, borderRadius: 20,
              border: '1px solid rgba(255,255,255,0.06)',
              background: 'rgba(10,10,10,0.9)',
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
              minHeight: 190,
            }}>
              <div style={{ position: 'absolute', right: -20, top: -20, opacity: 0.06 }}>
                <Database size={88} color="#a1a1aa" />
              </div>
              <div>
                <p style={{
                  fontSize: 10, fontWeight: 700, letterSpacing: '0.1em',
                  textTransform: 'uppercase', color: '#52525b', marginBottom: 14,
                }}>Scans Performed</p>
                <h3 className="stat-val" style={{
                  fontFamily: 'Syne, sans-serif', fontWeight: 800,
                  fontSize: 40, letterSpacing: '-0.04em', color: '#fff', margin: 0,
                }}>1,048</h3>
                <p style={{ fontSize: 12, color: '#52525b', marginTop: 6 }}>
                  Secured this session
                </p>
              </div>
              <div style={{
                width: '100%', height: 3, borderRadius: 10,
                background: 'rgba(255,255,255,0.05)', overflow: 'hidden',
              }}>
                <div style={{
                  height: '100%', borderRadius: 10,
                  width: '72%',
                  background: 'linear-gradient(90deg,#34d399,#22d3ee)',
                }} />
              </div>
            </div>

          </div>

          {/* ════ QUICK ACTIONS ROW ════ */}
          <div ref={addToRefs} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 12 }}>
            {[
              {
                icon: Camera, label: 'Live Detection',
                desc: 'Analyze webcam in real-time',
                action: () => navigate('/detector'),
                accent: '#34d399',
              },
              {
                icon: Layers, label: 'Video Upload',
                desc: 'Run inference on video files',
                action: () => navigate('/detector'),
                accent: '#22d3ee',
              },
              {
                icon: User, label: 'Image Scan',
                desc: 'Check static images instantly',
                action: () => navigate('/detector'),
                accent: '#818cf8',
              },
              {
                icon: Video, label: 'Live Interview',
                desc: 'Detect deepfakes during live calls',
                action: () => navigate('/live-call'),
                accent: '#c084fc',  // purple-400 — distinct from other cards
              },
            // eslint-disable-next-line no-unused-vars
            ].map(({ icon: Icon, label, desc, action, accent }) => (
              <button
                key={label}
                onClick={action}
                style={{
                  display: 'flex', alignItems: 'center', gap: 14,
                  padding: '16px 18px', borderRadius: 16, textAlign: 'left',
                  border: '1px solid rgba(255,255,255,0.05)',
                  background: 'rgba(255,255,255,0.02)',
                  cursor: 'pointer', color: '#fff',
                  transition: 'all 0.15s',
                  fontFamily: 'DM Sans, sans-serif',
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                  e.currentTarget.style.borderColor = `${accent}30`;
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.02)';
                  e.currentTarget.style.borderColor = 'rgba(255,255,255,0.05)';
                }}
              >
                <div style={{
                  width: 38, height: 38, borderRadius: 11, flexShrink: 0,
                  background: `${accent}12`,
                  border: `1px solid ${accent}20`,
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                }}>
                  <Icon size={15} color={accent} />
                </div>
                <div style={{ minWidth: 0 }}>
                  <p style={{ fontSize: 13, fontWeight: 600, lineHeight: 1, marginBottom: 4 }}>{label}</p>
                  <p style={{ fontSize: 11, color: '#52525b', lineHeight: 1.3 }}>{desc}</p>
                </div>
                <ChevronRight size={13} color="#3f3f46" style={{ marginLeft: 'auto', flexShrink: 0 }} />
              </button>
            ))}
          </div>

          {/* ════ RECENT SCANS ════ */}
          <div ref={addToRefs}>
            <div style={{
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              marginBottom: 14, paddingBottom: 14,
              borderBottom: '1px solid rgba(255,255,255,0.05)',
            }}>
              <h2 style={{
                fontFamily: 'Syne, sans-serif', fontWeight: 700,
                fontSize: 15, letterSpacing: '-0.02em', color: '#fff', margin: 0,
              }}>
                Recent Scans
              </h2>
              <button style={{
                fontSize: 11, color: '#34d399', fontWeight: 600,
                background: 'none', border: 'none', cursor: 'pointer',
                letterSpacing: '0.05em', fontFamily: 'DM Sans, sans-serif',
                display: 'flex', alignItems: 'center', gap: 4,
              }}>
                View All Logs
                <ChevronRight size={11} />
              </button>
            </div>

            {/* Empty state — attractive, not generic */}
            <div style={{
              position: 'relative', overflow: 'hidden',
              padding: '52px 28px', borderRadius: 20, textAlign: 'center',
              border: '1px dashed rgba(255,255,255,0.07)',
              background: 'rgba(255,255,255,0.01)',
            }}>
              {/* decorative grid lines */}
              <div style={{
                position: 'absolute', inset: 0, pointerEvents: 'none', opacity: 0.04,
                backgroundImage: 'linear-gradient(rgba(255,255,255,0.5) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,0.5) 1px, transparent 1px)',
                backgroundSize: '40px 40px',
                borderRadius: 20,
              }} />
              <div style={{
                width: 56, height: 56, borderRadius: 16, margin: '0 auto 16px',
                border: '1px solid rgba(255,255,255,0.07)',
                background: 'rgba(52,211,153,0.05)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
              }}>
                <BarChart3 size={22} color="#34d399" style={{ opacity: 0.6 }} />
              </div>
              <h4 style={{
                fontFamily: 'Syne, sans-serif', fontWeight: 700,
                fontSize: 15, color: '#d4d4d8', marginBottom: 8, margin: '0 0 8px',
              }}>
                No trace logs yet
              </h4>
              <p style={{ fontSize: 12, color: '#52525b', maxWidth: 320, margin: '0 auto 24px', lineHeight: 1.6 }}>
                Analyze a live stream, video, or image in the Detector Studio — your secure logs will appear here.
              </p>
              <button
                onClick={() => navigate('/detector')}
                className="btn-shimmer"
                style={{
                  display: 'inline-flex', alignItems: 'center', gap: 7,
                  padding: '10px 20px', borderRadius: 12,
                  border: 'none', color: '#080808',
                  fontSize: 12, fontWeight: 700,
                  cursor: 'pointer',
                  fontFamily: 'DM Sans, sans-serif',
                }}
              >
                <Zap size={12} />
                Analyze Media
              </button>
            </div>
          </div>

        </div>
      </main>
    </div>
  );
}
