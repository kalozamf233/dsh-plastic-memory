// client.js 语法与结构检查：mock window/require，真实执行 load 工厂
import { readFileSync } from 'node:fs'

const src = readFileSync(new URL('./lib/client.js', import.meta.url), 'utf8')

const react = {
  useEffect: () => {},
  useState: (init) => [init, () => {}],
  useRef: (init) => ({ current: init }),
}
const jsxRuntime = { jsx: (type, props) => ({ type, props }), jsxs: (type, props) => ({ type, props }) }
const fakeRequire = (id) => {
  if (id === 'react') return react
  if (id === 'react/jsx-runtime') return jsxRuntime
  throw new Error('unexpected require: ' + id)
}

let loaded = null
globalThis.window = { __ModuleLoader__: { load: (payload) => { loaded = payload } } }

// 1) 整文件语法 + load 调用
new Function('window', src)(globalThis.window)
if (!loaded) throw new Error('FAIL: __ModuleLoader__.load 未被调用')
if (loaded.id !== 'dsh-plastic-memory') throw new Error('FAIL: id 错误 ' + loaded.id)

// 2) 执行 factory 内部（mock require 走 fakeRequire，但 factory 内部 require 变量来自参数）
const module = { exports: {} }
const exports = module.exports
// factory 签名：factory(require) 使用闭包内的 module/exports/var 声明
// 直接调用 factory：内部 require 应使用传入参数——但源码里 require 未从参数取？
// 源码写法：factory: (require) => {...}，内部 require(...) 使用参数。OK。
const result = loaded.factory(fakeRequire)
console.log('client.js 检查通过')
console.log('  exports.name =', result && result.name)
console.log('  exports.inject =', JSON.stringify(result && result.inject))
console.log('  exports.apply 类型 =', typeof (result && result.apply))

// 3) 模拟 apply：slots mock
const slots = {
  inject: (name, fn) => {
    const reg = fn()
    console.log('  slots.inject(' + name + ') → 注册', reg.name, 'id/key =', reg.id ?? reg.key)
  },
  register: (spec, comp) => ({ ...spec, comp: typeof comp }),
}
const ctx = { slots }
result.apply(ctx)
console.log('apply 执行 OK')
