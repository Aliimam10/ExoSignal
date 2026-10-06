export function artifactUrl(relative: string) {
  return `/api/artifact/${relative.split("/").map(encodeURIComponent).join("/")}`;
}
