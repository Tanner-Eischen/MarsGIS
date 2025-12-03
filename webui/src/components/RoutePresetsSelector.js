import { jsx as _jsx } from "react/jsx-runtime";
import PresetsSelector from './PresetsSelector';
export default function RoutePresetsSelector({ presets, selected, onSelect }) {
    // Filter to route presets only
    const routePresets = presets.filter(p => p.scope === 'route');
    return _jsx(PresetsSelector, { presets: routePresets, selected: selected, onSelect: onSelect });
}
