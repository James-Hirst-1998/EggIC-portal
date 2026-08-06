/* PostHog (EU region), loaded by every page.
 *
 * Requests go to /ingest rather than eu.i.posthog.com directly: vercel.json
 * rewrites that path onto PostHog, so the calls look first-party and are not
 * dropped by the tracker blockers a fair share of visitors run. The project
 * token below is a public write-only key — it is meant to sit in the page.
 *
 * To go cookieless (no consent banner needed, but no returning-visitor counts),
 * flip COOKIELESS to true.
 */
(function () {
  var TOKEN = 'phc_D6iBgYsvnwC6Eiyc8B3ATNJMvhvvdRUSsmSacsXffiyR';
  var COOKIELESS = false;

  !function(t,e){var o,n,p,r;e.__SV||(window.posthog=e,e._i=[],e.init=function(i,s,a){function g(t,e){var o=e.split(".");2==o.length&&(t=t[o[0]],e=o[1]),t[e]=function(){t.push([e].concat(Array.prototype.slice.call(arguments,0)))}}(p=t.createElement("script")).type="text/javascript",p.crossOrigin="anonymous",p.async=!0,p.src=s.api_host.replace(".i.posthog.com","-assets.i.posthog.com")+"/static/array.js",(r=t.getElementsByTagName("script")[0]).parentNode.insertBefore(p,r);var u=e;for(void 0!==a?u=e[a]=[]:a="posthog",u.people=u.people||[],u.toString=function(t){var e="posthog";return"posthog"!==a&&(e+="."+a),t||(e+=" (stub)"),e},u.people.toString=function(){return u.toString(1)+".people (stub)"},o="init capture register register_once register_for_session unregister unregister_for_session getFeatureFlag getFeatureFlagPayload isFeatureEnabled reloadFeatureFlags updateEarlyAccessFeatureEnrollment getEarlyAccessFeatures on onFeatureFlags onSessionId getSurveys getActiveMatchingSurveys renderSurvey canRenderSurvey getNextSurveyStep identify setPersonProperties group resetGroups setPersonPropertiesForFlags resetPersonPropertiesForFlags setGroupPropertiesForFlags resetGroupPropertiesForFlags reset get_distinct_id getGroups get_session_id get_session_replay_url alias set_config startSessionRecording stopSessionRecording sessionRecordingStarted captureException loadToolbar get_property getSessionProperty createPersonProfile opt_in_capturing opt_out_capturing has_opted_in_capturing has_opted_out_capturing clear_opt_in_out_capturing debug getPageViewId captureTraceFeedback captureTraceMetric".split(" "),n=0;n<o.length;n++)g(u,o[n]);e._i.push([i,s,a])},e.__SV=1)}(document,window.posthog||[]);

  posthog.init(TOKEN, {
    api_host: '/ingest',
    ui_host: 'https://eu.posthog.com',

    // Anonymous visitors still get a person, so "how many people, from where,
    // coming back how often" is answerable — not just "how many events".
    person_profiles: 'always',
    persistence: COOKIELESS ? 'memory' : 'localStorage+cookie',

    capture_pageview: true,
    capture_pageleave: true,      // gives time-on-page
    autocapture: true,            // every click/submit, without naming each one
    capture_performance: true,    // page load timing
    capture_exceptions: true,     // uncaught JS errors land in Error tracking
  });

  posthog.register({
    app: 'eggic-portal',
    screen_class: window.innerWidth < 880 ? 'narrow' : 'wide',
    touch: ('ontouchstart' in window) || navigator.maxTouchPoints > 0,
  });

  /* Never let a blocked or failed analytics call break the page. */
  window.track = function (name, props) {
    try {
      if (window.posthog && typeof posthog.capture === 'function') {
        posthog.capture(name, props || {});
      }
    } catch (e) { /* ignore */ }
  };
})();
