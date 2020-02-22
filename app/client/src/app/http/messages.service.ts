import { Injectable } from '@angular/core';
import { HttpClient } from "@angular/common/http";

const API_BASE = "http://127.0.0.1:8000"
const API_MESSAGES = `${API_BASE}/api/classifier/messages/`


@Injectable({
    providedIn: 'root'
})
export class MessagesService {

    constructor(private http: HttpClient) { }

    
    // Classify a message (POST method)
    classifyMessage(params) {
        return this.http.post(API_MESSAGES, params)
    }

}
