import axios from "axios";

export const API = import.meta.env.VITE_API_URL;

// Two clients, because the calls fall into two very different shapes.
//
// `api` is for requests that should be quick — status polls, small GETs. A tight
// timeout here matters: without one, axios waits forever, so a request that will
// never be answered just stalls the UI silently.
//
// `apiLong` is for uploads and training, which legitimately take minutes on a
// small container. Giving these the short timeout would abort work that was
// about to succeed, which is worse than waiting.
export const api = axios.create({ baseURL: API, timeout: 20000 });
export const apiLong = axios.create({ baseURL: API, timeout: 10 * 60 * 1000 });

/** True when the failure is "no answer from the server", not "server said no".
 *
 * The distinction is the whole point: a 4xx is a definite answer and should be
 * shown to the user as-is, while a timeout or dropped connection is usually a
 * free-tier container waking up or restarting, and is worth retrying.
 */
export const isNetworkError = (err) =>
  !err?.response &&
  (err?.code === "ECONNABORTED" ||
    err?.code === "ERR_NETWORK" ||
    err?.message === "Network Error" ||
    err?.message?.includes("timeout"));

/** Human-readable text for any axios failure. */
export const errorMessage = (err, fallback = "Something went wrong.") => {
  const detail = err?.response?.data?.detail;
  if (typeof detail === "string") return detail;
  if (isNetworkError(err))
    return "The server is not responding. It may be starting up — wait a few seconds and try again.";
  if (err?.response?.status >= 500)
    return "The server hit an error handling that request. Try again with a smaller dataset.";
  return err?.message || fallback;
};

/** Wake a sleeping free-tier backend, retrying while it boots.
 *
 * A container that has been idle takes ~50s to answer its first request. Calling
 * this on app start means the wake happens while the user is still choosing a
 * file, instead of inside their upload where it looks like a hang.
 */
export const wakeBackend = async (attempts = 3) => {
  for (let i = 0; i < attempts; i++) {
    try {
      const res = await api.get("/health", { timeout: 30000 });
      return res.data;
    } catch {
      await new Promise((r) => setTimeout(r, 2000 * (i + 1)));
    }
  }
  return null;
};
