import feathers from "@feathersjs/feathers";
import rest from "@feathersjs/rest-client";

// import socketio from "@feathersjs/socketio-client";
// import io from "socket.io-client";
// import auth from "@feathersjs/authentication-client";

const url = import.meta.env.DEV
  ? "http://localhost:3030"
  : window.location.origin;
console.log("ApiUrl", url);

/***** REST *****/
export const api = feathers().configure(rest(url).fetch(fetch));

/******* SOCKETIO ******/
// const socket = io(url, { transports: ["websocket"] });

// This variable name becomes the alias for this server.
// export const api = feathers().configure(socketio(socket));
//   .configure(auth({ storage: window.localStorage }));
