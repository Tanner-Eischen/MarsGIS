import { Link, useLocation } from 'react-router-dom'
import { ReactNode } from 'react'
import { LayoutDashboard, Navigation, Database, FlaskConical, Menu, X } from 'lucide-react'

interface LayoutProps {
  children: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'ANALYSIS', icon: <LayoutDashboard size={18} /> },
    { path: '/mission', label: 'MISSION', icon: <Navigation size={18} /> },
    { path: '/decision-lab', label: 'DECISION', icon: <FlaskConical size={18} /> },
    { path: '/settings', label: 'DATA & CONFIG', icon: <Database size={18} /> },
  ]

  return (
    <div className="h-screen flex flex-col bg-black text-white font-sans overflow-hidden">
      {/* HUD Header */}
      <header className="h-16 bg-gray-900/80 backdrop-blur border-b border-cyan-900/30 flex items-center justify-between px-6 relative z-50">
        {/* Logo Area */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 bg-gradient-to-br from-mars-orange to-red-600 rounded-sm flex items-center justify-center shadow-lg shadow-orange-900/20">
            <span className="font-bold text-white text-lg">M</span>
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-widest text-white">MARS<span className="text-mars-orange">GIS</span></h1>
            <div className="text-[0.6rem] text-cyan-500 tracking-[0.2em] uppercase">Habitat Selection System</div>
          </div>
        </div>

        {/* Main Navigation */}
        <nav className="flex items-center gap-1 bg-gray-800/50 p-1 rounded-lg border border-gray-700/50">
          {navItems.map((item) => {
            const isActive = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path))
            return (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-2 px-4 py-2 rounded-md text-sm font-bold transition-all duration-200 ${
                  isActive
                    ? 'bg-cyan-900/30 text-cyan-400 border border-cyan-500/30 shadow-[0_0_10px_rgba(6,182,212,0.15)]'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
              >
                {item.icon}
                <span className="tracking-wider">{item.label}</span>
              </Link>
            )
          })}
        </nav>

        {/* Status / User (Placeholder) */}
        <div className="flex items-center gap-4">
            <div className="flex flex-col items-end">
                <span className="text-xs text-cyan-400 font-mono">SYS.ONLINE</span>
                <span className="text-[0.6rem] text-gray-500 font-mono">V.2.0.4</span>
            </div>
            <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]"></div>
        </div>
      </header>

      {/* Main Content Window */}
      <main className="flex-1 overflow-hidden relative bg-grid-pattern">
        {/* Children are rendered full height/width, controlling their own scrolling */}
        {children}
      </main>
    </div>
  )
}
