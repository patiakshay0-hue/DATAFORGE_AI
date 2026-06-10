import React, { createContext, useContext, useState, useEffect } from 'react'

const Ctx = createContext({ isDark: true, toggleTheme: () => {} })

export const ThemeProvider = ({ children }) => {
  const [isDark, setIsDark] = useState(
    () => localStorage.getItem('df-theme') !== 'light'
  )

  useEffect(() => {
    localStorage.setItem('df-theme', isDark ? 'dark' : 'light')
    document.documentElement.setAttribute('data-theme', isDark ? 'dark' : 'light')
  }, [isDark])

  return (
    <Ctx.Provider value={{ isDark, toggleTheme: () => setIsDark(v => !v) }}>
      {children}
    </Ctx.Provider>
  )
}

export const useTheme = () => useContext(Ctx)
