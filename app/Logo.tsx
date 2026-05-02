export function Logo() {
  return (
    <div className="flex items-center w-[192px] gap-2">
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        className="text-white"
      >
        <rect x="2" y="2" width="20" height="20" rx="4" stroke="currentColor" strokeWidth="2" />
        <polygon points="9,7 9,17 17,12" fill="currentColor" />
      </svg>
      <span className="font-bold text-white">VibeMatch</span>
    </div>
  );
}
